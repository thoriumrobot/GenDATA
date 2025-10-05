/*
    @Positive
 * Copyright (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.vm.ci.hotspot;

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
import static jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.runtime;
    @Positive
import static jdk.vm.ci.services.Services.IS_IN_NATIVE_IMAGE;
    @Positive
import jdk.vm.ci.meta.Assumptions;
    @Positive
import jdk.vm.ci.meta.JavaConstant;
    @Positive
import jdk.vm.ci.meta.JavaKind;
    @Positive
import jdk.vm.ci.meta.ResolvedJavaType;

    @Positive
abstract class HotSpotObjectConstantImpl implements HotSpotObjectConstant {

    @Positive
    protected final boolean compressed;

    @Positive
    @Override
    @Positive
    public JavaKind getJavaKind();

    @Positive
    @Override
    @Positive
    public boolean isCompressed();

    @Positive
    @Override
    @Positive
    public abstract JavaConstant compress();

    @Positive
    @Override
    @Positive
    public abstract JavaConstant uncompress();

    @Positive
    @Override
    @Positive
    public HotSpotResolvedObjectType getType();

    @Positive
    @Override
    @Positive
    public abstract int getIdentityHashCode();

    @Positive
    @Override
    @Positive
    public JavaConstant getCallSiteTarget(Assumptions assumptions);

    @Positive
    @Override
    @Positive
    public boolean isInternedString();

    @Positive
    @Override
    @Positive
    public <T> T asObject(Class<T> type);

    @Positive
    @Override
    @Positive
    public Object asObject(ResolvedJavaType type);

    @Positive
    @Override
    @Positive
    public boolean isNull();

    @Positive
    @Override
    @Positive
    public boolean isDefaultForKind();

    @Positive
    @Override
    @Positive
    public Object asBoxedPrimitive();

    @Positive
    @Override
    @Positive
    public int asInt();

    @Positive
    @Override
    @Positive
    public boolean asBoolean();

    @Positive
    @Override
    @Positive
    public long asLong();

    @Positive
    @Override
    @Positive
    public float asFloat();

    @Positive
    @Override
    @Positive
    public double asDouble();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    @Override
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toValueString();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public JavaConstant readFieldValue(HotSpotResolvedJavaField field, boolean isVolatile);

    @Positive
    public ResolvedJavaType asJavaType();
    @Positive
}

// CFWR semantic augmentation - variant 1
