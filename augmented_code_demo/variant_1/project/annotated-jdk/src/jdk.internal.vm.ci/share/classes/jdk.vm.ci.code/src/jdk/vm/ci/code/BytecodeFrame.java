/*
    @Positive
 * Copyright (c) 2009, 2015, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.
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
package jdk.vm.ci.code;

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
import java.util.Arrays;
    @Positive
import jdk.vm.ci.meta.JavaKind;
    @Positive
import jdk.vm.ci.meta.JavaValue;
    @Positive
import jdk.vm.ci.meta.ResolvedJavaMethod;
    @Positive
import jdk.vm.ci.meta.Value;

    @Positive
public final class BytecodeFrame extends BytecodePosition {

    @Positive
    @SuppressFBWarnings(value = "EI_EXPOSE_REP2", justification = "field is intentionally mutable")
    @Positive
    public final JavaValue[] values;

    @Positive
    public final int numLocals;

    @Positive
    public final int numStack;

    @Positive
    public final int numLocks;

    @Positive
    public final boolean rethrowException;

    @Positive
    public final boolean duringCall;

    @Positive
    public static final int UNKNOWN_BCI;

    @Positive
    public static final int UNWIND_BCI;

    @Positive
    public static final int BEFORE_BCI;

    @Positive
    public static final int AFTER_BCI;

    @Positive
    public static final int AFTER_EXCEPTION_BCI;

    @Positive
    public static final int INVALID_FRAMESTATE_BCI;

    @Positive
    public static boolean isPlaceholderBci(int bci);

    @Positive
    public static String getPlaceholderBciName(int bci);

    @Positive
    @SuppressFBWarnings(value = "EI_EXPOSE_REP2", justification = "caller transfers ownership of `slotKinds`")
    @Positive
    public BytecodeFrame(BytecodeFrame caller, ResolvedJavaMethod method, int bci, boolean rethrowException, boolean duringCall, JavaValue[] values, JavaKind[] slotKinds, int numLocals, int numStack, int numLocks) {
    @Positive
    }

    @Positive
    public boolean validateFormat();

    @Positive
    public JavaKind getLocalValueKind(int i);

    @Positive
    public JavaKind getStackValueKind(int i);

    @Positive
    public JavaValue getLocalValue(int i);

    @Positive
    public JavaValue getStackValue(int i);

    @Positive
    public JavaValue getLockValue(int i);

    @Positive
    public BytecodeFrame caller();

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
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
