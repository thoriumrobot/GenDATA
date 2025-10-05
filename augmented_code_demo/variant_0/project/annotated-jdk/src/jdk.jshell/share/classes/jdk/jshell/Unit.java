/*
    @Positive
 * Copyright (c) 2015, 2016, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package jdk.jshell;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.jshell.ClassTracker.ClassInfo;
    @Positive
import jdk.jshell.Snippet.Kind;
    @Positive
import jdk.jshell.Snippet.Status;
    @Positive
import jdk.jshell.Snippet.SubKind;
    @Positive
import jdk.jshell.TaskFactory.AnalyzeTask;
    @Positive
import jdk.jshell.TaskFactory.CompileTask;
    @Positive
import jdk.jshell.spi.ExecutionControl.ClassBytecodes;
    @Positive
import jdk.jshell.spi.ExecutionControl.ClassInstallException;
    @Positive
import jdk.jshell.spi.ExecutionControl.EngineTerminationException;
    @Positive
import jdk.jshell.spi.ExecutionControl.NotImplementedException;
    @Positive
import static java.util.stream.Collectors.toSet;
    @Positive
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_EVNT;
    @Positive
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_GEN;
    @Positive
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_WRAP;
    @Positive
import static jdk.jshell.Snippet.Status.OVERWRITTEN;
    @Positive
import static jdk.jshell.Snippet.Status.RECOVERABLE_DEFINED;
    @Positive
import static jdk.jshell.Snippet.Status.RECOVERABLE_NOT_DEFINED;
    @Positive
import static jdk.jshell.Snippet.Status.REJECTED;
    @Positive
import static jdk.jshell.Snippet.Status.VALID;
    @Positive
import static jdk.jshell.Util.PARSED_LOCALE;
    @Positive
import static jdk.jshell.Util.expunge;

    @Positive
final class Unit {

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    Snippet snippet();

    @Positive
    boolean isDependency();

    @Positive
    void initialize();

    @Positive
    void setWrap(Collection<Unit> exceptUnit, Collection<Unit> plusUnfiltered);

    @Positive
    void setDiagnostics(AnalyzeTask ct);

    @Positive
    void setDiagnostics(DiagList diags);

    @Positive
    boolean corralIfNeeded(Collection<Unit> working);

    @Positive
    void setCorralledDiagnostics(AnalyzeTask cct);

    @Positive
    boolean smashingErrorDiagnostics(CompileTask ct);

    @Positive
    void setStatus(AnalyzeTask at);

    @Positive
    boolean isDefined();

    @Positive
    Stream<ClassBytecodes> classesToLoad(List<String> classnames);

    @Positive
    boolean doRedefines();

    @Positive
    void markForReplacement();

    @Positive
    Stream<Unit> effectedDependents();

    @Positive
    Stream<Unit> dependents();

    @Positive
    void finish();

    @Positive
    SnippetEvent event(String value, JShellException exception);

    @Positive
    List<SnippetEvent> secondaryEvents();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    private static class UnresolvedExtractor {

    @Positive
        DiagList otherCorralledErrors();

    @Positive
        DiagList otherAll();

    @Positive
        List<String> unresolved();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
