/*
 * Copyright (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.vm.ci.hotspot;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.runtime;
import static jdk.vm.ci.services.Services.IS_IN_NATIVE_IMAGE;
import jdk.vm.ci.meta.Assumptions;
import jdk.vm.ci.meta.JavaConstant;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.ResolvedJavaType;

abstract class HotSpotObjectConstantImpl implements HotSpotObjectConstant {

    protected final boolean compressed;

    @Override
    public JavaKind getJavaKind();

    @Override
    public boolean isCompressed();

    @Override
    public abstract JavaConstant compress();

    @Override
    public abstract JavaConstant uncompress();

    @Override
    public HotSpotResolvedObjectType getType();

    @Override
    public abstract int getIdentityHashCode();

    @Override
    public JavaConstant getCallSiteTarget(Assumptions assumptions);

    @Override
    public boolean isInternedString();

    @Override
    public <T> T asObject(Class<T> type);

    @Override
    public Object asObject(ResolvedJavaType type);

    @Override
    public boolean isNull();

    @Override
    public boolean isDefaultForKind();

    @Override
    public Object asBoxedPrimitive();

    @Override
    public int asInt();

    @Override
    public boolean asBoolean();

    @Override
    public long asLong();

    @Override
    public float asFloat();

    @Override
    public double asDouble();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Override
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    @Override
    public String toValueString();

    @Override
    public String toString();

    public JavaConstant readFieldValue(HotSpotResolvedJavaField field, boolean isVolatile);

    public ResolvedJavaType asJavaType();
}
