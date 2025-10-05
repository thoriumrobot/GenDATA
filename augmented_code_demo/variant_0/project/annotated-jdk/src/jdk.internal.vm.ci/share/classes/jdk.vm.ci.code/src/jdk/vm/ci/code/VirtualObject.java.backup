/*
 * Copyright (c) 2010, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import jdk.vm.ci.common.JVMCIError;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.JavaValue;
import jdk.vm.ci.meta.ResolvedJavaField;
import jdk.vm.ci.meta.ResolvedJavaType;

public final class VirtualObject implements JavaValue {

    public static VirtualObject get(ResolvedJavaType type, int id);

    public static VirtualObject get(ResolvedJavaType type, int id, boolean isAutoBox);

    public interface LayoutVerifier {

        int getOffset(ResolvedJavaField field);

        default JavaKind getStorageKind(ResolvedJavaField field);
    }

    public void verifyLayout(LayoutVerifier verifier);

    @Override
    public String toString();

    public ResolvedJavaType getType();

    @SuppressFBWarnings(value = "EI_EXPOSE_REP", justification = "`values` is intentional mutable")
    public JavaValue[] getValues();

    public JavaKind getSlotKind(int index);

    public int getId();

    public boolean isAutoBox();

    @SuppressFBWarnings(value = "EI_EXPOSE_REP2", justification = "caller transfers ownership of `slotKinds`")
    public void setValues(JavaValue[] values, JavaKind[] slotKinds);

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);
}
