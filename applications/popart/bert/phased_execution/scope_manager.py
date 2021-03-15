# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains utils to setup scope for a Block."""
import inspect
import logging
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, List, Optional

import popart

logger = logging.getLogger(__name__)


class Scope:
    def __init__(self,
                 name: str,
                 vgid: Optional[int] = None,
                 execution_phase: Optional[int] = None,
                 pipeline_stage: Optional[int] = None,
                 additional_scopes: Optional[List]=None):

        if (execution_phase is not None) and (pipeline_stage is not None):
            raise ValueError(
                'Cannot set both `execution_phase` and `pipeline_stage`.')

        if execution_phase is not None:
            self.execution_phase: int = execution_phase
        if pipeline_stage is not None:
            self.pipeline_stage: int = pipeline_stage

        self.name: str = name
        self.vgid: int = vgid
        self.additional_scopes = additional_scopes

    def __setattr__(self, name: str, value: Any):
        if name == 'execution_phase':
            if hasattr(self, 'pipeline_stage'):
                raise ValueError(
                    'Cannot set `execution_phase` when `pipeline_stage` exists.'
                )
        if name == 'pipeline_stage':
            if hasattr(self, 'execution_phase'):
                raise ValueError(
                    'Cannot set `pipeline_stage` when `execution_phase` exists.'
                )
        super(Scope, self).__setattr__(name, value)

    def __repr__(self) -> str:
        if hasattr(self, 'execution_phase'):
            return f'Namescope: {self.name}, Execution phase: {self.execution_phase}, VGID: {self.vgid}'
        elif hasattr(self, 'pipeline_stage'):
            return f'Namescope: {self.name}, Pipeline stage: {self.pipeline_stage}, VGID: {self.vgid}'
        if hasattr(self, 'vgid'):
            return f'Namescope: {self.name}, VGID: {self.vgid}'
        return f'Namescope: {self.name}'


class ScopeProvider():
    """Utility that book-keeps scopes.

    This class tracks scope index assignment used in phased_execution mode.

    Attributes:
        phased_execution_type: 'DUAL' or 'SINGLE' denoting number of devices per replica in phased execution mode.
    """
    def __init__(self, phased_execution_type="DUAL"):
        if phased_execution_type == "DUAL":
            self.scope_increment = 1
            start_phase = -1
        else:
            self.scope_increment = 4
            start_phase = -4

        self.prev_phase = start_phase
        self.phased_execution_type = phased_execution_type

    def scope_provider(self, builder: popart.Builder,
                       scope: Scope) -> Iterator[ExitStack]:
        """Generate scope for popart layers.

        Args:
            builder (popart.Builder): Builder used to create popart model.
            scope: Scope

        Yields:
            Iterator[ExitStack]: Stack of builder contexts.
        """
        context = ExitStack()
        if hasattr(scope, 'name'):
            context.enter_context(builder.nameScope(scope.name))

        if hasattr(scope, 'execution_phase'):
            if (scope.execution_phase - self.prev_phase) > 1:
                logger.warning('Skipping phased_execution scope: {0} -> {1}'.format(
                    self.prev_phase, scope.execution_phase))

            context.enter_context(builder.executionPhase(scope.execution_phase))
            self.prev_phase = max(self.prev_phase, scope.execution_phase)

        if hasattr(scope, 'pipeline_stage'):
            context.enter_context(builder.pipelineStage(scope.pipeline_stage))

        if scope.vgid is not None:
            context.enter_context(builder.virtualGraph(scope.vgid))

        if scope.additional_scopes:
            for scope in scope.additional_scopes:
                context.enter_context(scope)

        return context

    @contextmanager
    def __call__(self, builder, scope):
        logger.debug(scope)
        context = self.scope_provider(builder, scope)
        yield context
        context.close()

    def get_next_phase(self) -> int:
        """Get next execution phase.

        Returns:
            int: Next execution phase.
        """
        self.prev_phase += self.scope_increment
        return self.prev_phase

    def get_prev_phase(self) -> int:
        """Get last execution phase.

        Returns:
            int: Previous execution phase.
        """
        return self.prev_phase

    def get_scope(self, name, execution_phase=None, skip_scope=False, additional_scopes=None):
        if inspect.stack()[1].function == 'forward':
            raise ValueError(
                'Scoping must be assigned during layer definition, before the forward pass.'
            )
        if execution_phase is None:
            return Scope(name, additional_scopes=additional_scopes)

        if execution_phase == 'next':
            if skip_scope:
                self.get_next_phase()
            execution_phase = self.get_next_phase()

        if execution_phase == 'prev':
            execution_phase = self.get_prev_phase()

        vgid = execution_phase % 2

        return Scope(name, vgid, execution_phase, additional_scopes=additional_scopes)
