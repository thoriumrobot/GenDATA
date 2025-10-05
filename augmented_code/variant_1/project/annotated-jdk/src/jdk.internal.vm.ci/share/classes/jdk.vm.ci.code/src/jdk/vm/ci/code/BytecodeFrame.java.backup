/*
 * Copyright (c) 2009, 2015, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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
package jdk.vm.ci.code;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Arrays;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.JavaValue;
import jdk.vm.ci.meta.ResolvedJavaMethod;
import jdk.vm.ci.meta.Value;

public final class BytecodeFrame extends BytecodePosition {

    @SuppressFBWarnings(value = "EI_EXPOSE_REP2", justification = "field is intentionally mutable")
    public final JavaValue[] values;

    public final int numLocals;

    public final int numStack;

    public final int numLocks;

    public final boolean rethrowException;

    public final boolean duringCall;

    public static final int UNKNOWN_BCI;

    public static final int UNWIND_BCI;

    public static final int BEFORE_BCI;

    public static final int AFTER_BCI;

    public static final int AFTER_EXCEPTION_BCI;

    public static final int INVALID_FRAMESTATE_BCI;

    public static boolean isPlaceholderBci(int bci);

    public static String getPlaceholderBciName(int bci);

    @SuppressFBWarnings(value = "EI_EXPOSE_REP2", justification = "caller transfers ownership of `slotKinds`")
    public BytecodeFrame(BytecodeFrame caller, ResolvedJavaMethod method, int bci, boolean rethrowException, boolean duringCall, JavaValue[] values, JavaKind[] slotKinds, int numLocals, int numStack, int numLocks) {
    }

    public boolean validateFormat();

    public JavaKind getLocalValueKind(int i);

    public JavaKind getStackValueKind(int i);

    public JavaValue getLocalValue(int i);

    public JavaValue getStackValue(int i);

    public JavaValue getLockValue(int i);

    public BytecodeFrame caller();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public String toString();
}
