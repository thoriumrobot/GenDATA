/*
 * Copyright (c) 2015, 2016, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package jdk.jshell;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import jdk.jshell.ClassTracker.ClassInfo;
import jdk.jshell.Snippet.Kind;
import jdk.jshell.Snippet.Status;
import jdk.jshell.Snippet.SubKind;
import jdk.jshell.TaskFactory.AnalyzeTask;
import jdk.jshell.TaskFactory.CompileTask;
import jdk.jshell.spi.ExecutionControl.ClassBytecodes;
import jdk.jshell.spi.ExecutionControl.ClassInstallException;
import jdk.jshell.spi.ExecutionControl.EngineTerminationException;
import jdk.jshell.spi.ExecutionControl.NotImplementedException;
import static java.util.stream.Collectors.toSet;
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_EVNT;
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_GEN;
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_WRAP;
import static jdk.jshell.Snippet.Status.OVERWRITTEN;
import static jdk.jshell.Snippet.Status.RECOVERABLE_DEFINED;
import static jdk.jshell.Snippet.Status.RECOVERABLE_NOT_DEFINED;
import static jdk.jshell.Snippet.Status.REJECTED;
import static jdk.jshell.Snippet.Status.VALID;
import static jdk.jshell.Util.PARSED_LOCALE;
import static jdk.jshell.Util.expunge;

final class Unit {

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    Snippet snippet();

    boolean isDependency();

    void initialize();

    void setWrap(Collection<Unit> exceptUnit, Collection<Unit> plusUnfiltered);

    void setDiagnostics(AnalyzeTask ct);

    void setDiagnostics(DiagList diags);

    boolean corralIfNeeded(Collection<Unit> working);

    void setCorralledDiagnostics(AnalyzeTask cct);

    boolean smashingErrorDiagnostics(CompileTask ct);

    void setStatus(AnalyzeTask at);

    boolean isDefined();

    Stream<ClassBytecodes> classesToLoad(List<String> classnames);

    boolean doRedefines();

    void markForReplacement();

    Stream<Unit> effectedDependents();

    Stream<Unit> dependents();

    void finish();

    SnippetEvent event(String value, JShellException exception);

    List<SnippetEvent> secondaryEvents();

    @Override
    public String toString();

    private static class UnresolvedExtractor {

        DiagList otherCorralledErrors();

        DiagList otherAll();

        List<String> unresolved();
    }
}
