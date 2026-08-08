from __future__ import annotations

import ast
import json
import os
import logging
import textwrap
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)
import time

from openai import (
    AsyncStream,
    Stream,
)
from pydantic import BaseModel, ValidationError

from camel.agents._types import ModelResponse, ToolCallRequest
from camel.agents._utils import (
    convert_to_function_tool,
    convert_to_schema,
    get_info_dict,
    handle_logprobs,
    safe_model_dump,
)
from camel.agents.base import BaseAgent
from camel.memories import (
    AgentMemory,
    ChatHistoryMemory,
    MemoryRecord,
    ScoreBasedContextCreator,
)
from camel.messages import BaseMessage, FunctionCallingMessage, OpenAIMessage
from camel.models import (
    BaseModelBackend,
    ModelFactory,
    ModelManager,
    ModelProcessingError,
)
from camel.prompts import TextPrompt
from camel.responses import ChatAgentResponse
from camel.toolkits import FunctionTool
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelPlatformType,
    ModelType,
    OpenAIBackendRole,
    RoleType,
)
from camel.types.agents import ToolCallingRecord
from camel.utils import (
    get_model_encoding,
    func_string_to_callable,
    get_pydantic_object_schema,
    json_to_function_code,
)
from camel.agents.chat_agent import ChatAgent
from retry import retry
import openai

if TYPE_CHECKING:
    from camel.terminators import ResponseTerminator


logger = logging.getLogger(__name__)


def _proxy_on():
    os.environ["http_proxy"] = "http://star-proxy.oa.com:3128"
    os.environ["https_proxy"] = "http://star-proxy.oa.com:3128"
    
def _proxy_off():
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""


class OwlChatAgent(ChatAgent):
    def __init__(
        self, 
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[
            Union[BaseModelBackend, List[BaseModelBackend]]
        ] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: Optional[int] = None,
        token_limit: Optional[int] = None,
        output_language: Optional[str] = None,
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        external_tools: Optional[
            List[Union[FunctionTool, Callable, Dict[str, Any]]]
        ] = None,
        response_terminators: Optional[List[ResponseTerminator]] = None,
        scheduling_strategy: str = "round_robin",
        single_iteration: bool = False,
        agent_id: Optional[str] = None,
    ):
        super().__init__(
            system_message,
            model,
            memory,
            message_window_size,
            token_limit,
            output_language,
            tools,
            external_tools,
            response_terminators,
            scheduling_strategy,
            single_iteration,
            agent_id
        )

    
    @retry(openai.APIConnectionError, backoff=2, max_delay=60)
    def step(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15,
        tool_call_based_structured_output: Optional[bool] = True,
    ) -> ChatAgentResponse:
        
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Add user input to memory
        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        # If tool_call_based_structured_output is True and we have a
        # response_format, add the output schema as a special tool
        if tool_call_based_structured_output and response_format:
            # Extract the schema from the response format and create a function
            schema_json = get_pydantic_object_schema(response_format)
            func_str = json_to_function_code(schema_json)
            func_callable = func_string_to_callable(func_str)

            # Create a function tool and add it to tools
            func_tool = FunctionTool(func_callable)
            self._internal_tools[
                self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
            ] = func_tool

        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            # Get response from model backend
            response = self._get_model_response(
                openai_messages,
                num_tokens,
                None if tool_call_based_structured_output else response_format,
                self._get_full_tool_schemas(),
            )

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_records.append(self._execute_tool(tool_call_request))
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break

                # For tool_call_based_structured_output, check if we need to
                # add the output schema after all tool calls are done but
                # before the final response
                if tool_call_based_structured_output and response_format:
                    # Determine if we need to update with structured output
                    # Check if all tool calls are not for the special
                    # structured output
                    if all(
                        record.tool_name
                        != self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records
                    ):
                        # Continue the loop to get a structured response
                        if not self.single_iteration:
                            continue

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            # If tool_call_based_structured_output and we have a response_format
            # but no tool calls were made for the structured output, we need to continue
            if (tool_call_based_structured_output and response_format and 
                not any(record.tool_name == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records) and
                not self.single_iteration):
                # Original owl re-prompts here ("please invoke the function...")
                # and loops. Parse what the model already returned instead.
                raw = response.output_messages[0].content.strip() if response.output_messages else ""
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                if parsed is None or not isinstance(parsed, dict):
                    parsed = {"content": raw, "failed": False}
                for message in response.output_messages:
                    message.content = json.dumps(parsed)
                break
                
            break

        # If using tool_call_based_structured_output and response_format is
        # provided, update the message content with the structured result
        if tool_call_based_structured_output and response_format:
            # Go through tool calls and process any special structured output
            # calls
            for record in tool_call_records:
                if (
                    record.tool_name
                    == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                ):
                    # Update all output messages with the structured output
                    # result
                    for message in response.output_messages:
                        message.content = str(record.result)
                    break
        # If not using tool call based structured output, format the response
        # if needed
        else:
            self._format_response_if_needed(response, response_format)
            
        self._record_final_output(response.output_messages)
        
        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            
            response.output_messages[0].content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""
            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )
        
        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )
        
        
    async def astep(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15,
        tool_call_based_structured_output: Optional[bool] = True,
    ) -> ChatAgentResponse:

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        # If tool_call_based_structured_output is True and we have a
        # response_format, add the output schema as a special tool
        if tool_call_based_structured_output and response_format:
            # Extract the schema from the response format and create a function
            schema_json = get_pydantic_object_schema(response_format)
            func_str = json_to_function_code(schema_json)
            func_callable = func_string_to_callable(func_str)

            # Create a function tool and add it to tools
            func_tool = FunctionTool(func_callable)
            self._internal_tools[
                self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
            ] = func_tool
            
        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            response = await self._aget_model_response(
                openai_messages,
                num_tokens,
                None if tool_call_based_structured_output else response_format,
                self._get_full_tool_schemas(),
            )

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_record = await self._aexecute_tool(tool_call_request)
                        tool_call_records.append(tool_call_record)
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break
                    
                # For tool_call_based_structured_output, check if we need to
                # add the output schema after all tool calls are done but
                # before the final response
                if tool_call_based_structured_output and response_format:
                    # Determine if we need to update with structured output
                    # Check if all tool calls are not for the special
                    # structured output
                    if all(
                        record.tool_name
                        != self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records
                    ):
                        # Continue the loop to get a structured response
                        if not self.single_iteration:
                            continue

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            # If tool_call_based_structured_output and we have a response_format
            # but no tool calls were made for the structured output, we need to continue
            if (tool_call_based_structured_output and response_format and 
                not any(record.tool_name == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records) and
                not self.single_iteration):
                # Original owl re-prompts here ("please invoke the function...")
                # and loops. Parse what the model already returned instead.
                raw = response.output_messages[0].content.strip() if response.output_messages else ""
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                if parsed is None or not isinstance(parsed, dict):
                    parsed = {"content": raw, "failed": False}
                for message in response.output_messages:
                    message.content = json.dumps(parsed)
                break
                
            break

        # If using tool_call_based_structured_output and response_format is
        # provided, update the message content with the structured result
        if tool_call_based_structured_output and response_format:
            # Go through tool calls and process any special structured output
            # calls
            for record in tool_call_records:
                if (
                    record.tool_name
                    == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                ):
                    # Update all output messages with the structured output
                    # result
                    for message in response.output_messages:
                        message.content = str(record.result)
                    break
        # If not using tool call based structured output, format the response
        # if needed
        else:
            await self._aformat_response_if_needed(response, response_format)
            
        self._record_final_output(response.output_messages)
        
        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            debug_content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""         
            response.output_messages[0].content = debug_content

            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )

        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )



class OwlWorkforceChatAgent(ChatAgent):
    def __init__(
        self, 
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[
            Union[BaseModelBackend, List[BaseModelBackend]]
        ] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: Optional[int] = None,
        token_limit: Optional[int] = None,
        output_language: Optional[str] = None,
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        external_tools: Optional[
            List[Union[FunctionTool, Callable, Dict[str, Any]]]
        ] = None,
        response_terminators: Optional[List[ResponseTerminator]] = None,
        scheduling_strategy: str = "round_robin",
        single_iteration: bool = False,
        agent_id: Optional[str] = None,
    ):
        super().__init__(
            system_message,
            model,
            memory,
            message_window_size,
            token_limit,
            output_language,
            tools,
            external_tools,
            response_terminators,
            scheduling_strategy,
            single_iteration,
            agent_id
        )
    
    
    @retry(openai.APIConnectionError, backoff=2, max_delay=60)
    def step(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15,
        tool_call_based_structured_output: Optional[bool] = True,
    ) -> ChatAgentResponse:
        
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Add user input to memory
        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        # If tool_call_based_structured_output is True and we have a
        # response_format, add the output schema as a special tool
        if tool_call_based_structured_output and response_format:
            # Extract the schema from the response format and create a function
            schema_json = get_pydantic_object_schema(response_format)
            func_str = json_to_function_code(schema_json)
            func_callable = func_string_to_callable(func_str)

            # Create a function tool and add it to tools
            func_tool = FunctionTool(func_callable)
            self._internal_tools[
                self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
            ] = func_tool

        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            # Get response from model backend
            A = time.time()
            response = self._get_model_response(
                openai_messages,
                num_tokens,
                None if tool_call_based_structured_output else response_format,
                self._get_full_tool_schemas(),
            )
            B = time.time()
            print("Elapsed Time Run Enhanced Agent sync (sec): ", B-A)
            print("Request IN: ", openai_messages, num_tokens)
            print("Answer OUT1: ", response)

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_records.append(self._execute_tool(tool_call_request))
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break

                # For tool_call_based_structured_output, check if we need to
                # add the output schema after all tool calls are done but
                # before the final response
                if tool_call_based_structured_output and response_format:
                    # Determine if we need to update with structured output
                    # Check if all tool calls are not for the special
                    # structured output
                    if all(
                        record.tool_name
                        != self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records
                    ):
                        # Continue the loop to get a structured response
                        if not self.single_iteration:
                            continue

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            # If tool_call_based_structured_output and we have a response_format
            # but no tool calls were made for the structured output, we need to continue
            if (tool_call_based_structured_output and response_format and 
                not any(record.tool_name == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records) and
                not self.single_iteration):
                # Original owl re-prompts here ("please invoke the function...")
                # and loops. Parse what the model already returned instead.
                raw = response.output_messages[0].content.strip() if response.output_messages else ""
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                if parsed is None or not isinstance(parsed, dict):
                    parsed = {"content": raw, "failed": False}
                for message in response.output_messages:
                    message.content = json.dumps(parsed)
                break
                
            break

        # If using tool_call_based_structured_output and response_format is
        # provided, update the message content with the structured result
        if tool_call_based_structured_output and response_format:
            # Go through tool calls and process any special structured output
            # calls
            for record in tool_call_records:
                if (
                    record.tool_name
                    == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                ):
                    # Update all output messages with the structured output
                    # result
                    for message in response.output_messages:
                        message.content = str(record.result)
                    break
        # If not using tool call based structured output, format the response
        # if needed
        else:
            self._format_response_if_needed(response, response_format)
            
        self._record_final_output(response.output_messages)
        
        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            debug_content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""
            # the content should be a json object
            response.output_messages[0].content = f"""
{{
    "content": "{debug_content}",
    "failed": true
}}
"""

            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )
        
        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )
    
    
    async def astep(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15,
        tool_call_based_structured_output: Optional[bool] = True,
    ) -> ChatAgentResponse:
        r"""Performs a single step in the chat session by generating a response
        to the input message. This agent step can call async function calls.

        Args:
            input_message (Union[BaseMessage, str]): The input message to the
                agent. For BaseMessage input, its `role` field that specifies
                the role at backend may be either `user` or `assistant` but it
                will be set to `user` anyway since for the self agent any
                incoming message is external. For str input, the `role_name`
                would be `User`.
            response_format (Optional[Type[BaseModel]], optional): A pydantic
                model class that includes value types and field descriptions
                used to generate a structured response by LLM. This schema
                helps in defining the expected output format. (default:
                :obj:`None`)
            max_tool_calls (int, optional): Maximum number of tool calls allowed
                before interrupting the process. (default: :obj:`15`)
            tool_call_based_structured_output (Optional[bool], optional): If
                True, uses tool calls to implement structured output. This
                approach treats the output schema as a special tool. (default:
                :obj:`False`)

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        # If tool_call_based_structured_output is True and we have a
        # response_format, add the output schema as a special tool
        if tool_call_based_structured_output and response_format:
            # Extract the schema from the response format and create a function
            schema_json = get_pydantic_object_schema(response_format)
            func_str = json_to_function_code(schema_json)
            func_callable = func_string_to_callable(func_str)

            # Create a function tool and add it to tools
            func_tool = FunctionTool(func_callable)
            self._internal_tools[
                self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
            ] = func_tool
        
        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )

            A = time.time()
            response = await self._aget_model_response(
                openai_messages,
                num_tokens,
                None if tool_call_based_structured_output else response_format,
                self._get_full_tool_schemas(),
            )
            B = time.time()
            print("Elapsed Time Run Enhanced Agent async (sec): ", B-A)
            print("Request IN: ", openai_messages, num_tokens)
            if response.response.choices[0].logprobs.content is not None:
                print(f"Answer OUT2: {len(response.response.choices[0].logprobs.content)} {response.response.model}\n", response.response.choices[0].message)
                print([t.token for t in response.response.choices[0].logprobs.content])
                print(response.usage_dict,'\n')
            else:
                print(f"Answer OUT22: {response.response.model}\n", response)
                print(response.usage_dict,'\n')


            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        AA = time.time()
                        tool_call_record = await self._aexecute_tool(tool_call_request)
                        BB = time.time()
                        print("Elapsed Time Tool Call Async (sec): ", BB-AA)
                        tool_call_records.append(tool_call_record)
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break
                # print("AAA")

                # For tool_call_based_structured_output, check if we need to
                # add the output schema after all tool calls are done but
                # before the final response
                if tool_call_based_structured_output and response_format:
                    # Determine if we need to update with structured output
                    # Check if all tool calls are not for the special
                    # structured output
                    if all(
                        record.tool_name
                        != self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records
                    ):
                        # Continue the loop to get a structured response
                        if not self.single_iteration:
                            continue

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            # If tool_call_based_structured_output and we have a response_format
            # but no tool calls were made for the structured output, we need to continue
            if (tool_call_based_structured_output and response_format and 
                not any(record.tool_name == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                        for record in tool_call_records) and
                not self.single_iteration):
                # Original owl re-prompts here ("please invoke the function...")
                # and loops. Parse what the model already returned instead.
                raw = response.output_messages[0].content.strip() if response.output_messages else ""
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                if parsed is None or not isinstance(parsed, dict):
                    parsed = {"content": raw, "failed": False}
                for message in response.output_messages:
                    message.content = json.dumps(parsed)
                break
                
            break

        # If using tool_call_based_structured_output and response_format is
        # provided, update the message content with the structured result
        if tool_call_based_structured_output and response_format:
            # Go through tool calls and process any special structured output
            # calls
            for record in tool_call_records:
                if (
                    record.tool_name
                    == self.__class__.Constants.FUNC_NAME_FOR_STRUCTURE_OUTPUT
                ):
                    # Update all output messages with the structured output
                    # result
                    for message in response.output_messages:
                        message.content = str(record.result)
                    break
        # If not using tool call based structured output, format the response
        # if needed
        else:
            await self._aformat_response_if_needed(response, response_format)

        self._record_final_output(response.output_messages)

        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            debug_content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""         
            # request should be a json object
            response_dict = {
                "content": debug_content,
                "failed": True
            }
            response.output_messages[0].content = json.dumps(response_dict)
            
            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )
            
        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )
        