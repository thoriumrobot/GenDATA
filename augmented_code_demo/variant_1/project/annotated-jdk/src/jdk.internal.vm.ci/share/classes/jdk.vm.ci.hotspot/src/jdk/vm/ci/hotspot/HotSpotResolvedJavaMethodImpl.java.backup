/*
 * Copyright (c) 2011, 2019, Oracle and/or its affiliates. All rights reserved.
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
import static jdk.vm.ci.hotspot.CompilerToVM.compilerToVM;
import static jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.runtime;
import static jdk.vm.ci.hotspot.HotSpotModifiers.BRIDGE;
import static jdk.vm.ci.hotspot.HotSpotModifiers.SYNTHETIC;
import static jdk.vm.ci.hotspot.HotSpotModifiers.VARARGS;
import static jdk.vm.ci.hotspot.HotSpotModifiers.jvmMethodModifiers;
import static jdk.vm.ci.hotspot.HotSpotVMConfig.config;
import static jdk.vm.ci.hotspot.UnsafeAccess.UNSAFE;
import java.lang.annotation.Annotation;
import java.lang.reflect.Executable;
import java.lang.reflect.Modifier;
import java.lang.reflect.Type;
import jdk.vm.ci.common.JVMCIError;
import jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.Option;
import jdk.vm.ci.meta.Constant;
import jdk.vm.ci.meta.ConstantPool;
import jdk.vm.ci.meta.DefaultProfilingInfo;
import jdk.vm.ci.meta.ExceptionHandler;
import jdk.vm.ci.meta.JavaMethod;
import jdk.vm.ci.meta.JavaType;
import jdk.vm.ci.meta.LineNumberTable;
import jdk.vm.ci.meta.Local;
import jdk.vm.ci.meta.LocalVariableTable;
import jdk.vm.ci.meta.ProfilingInfo;
import jdk.vm.ci.meta.ResolvedJavaMethod;
import jdk.vm.ci.meta.ResolvedJavaType;
import jdk.vm.ci.meta.SpeculationLog;
import jdk.vm.ci.meta.TriState;

final class HotSpotResolvedJavaMethodImpl extends HotSpotMethod implements HotSpotResolvedJavaMethod, MetaspaceHandleObject {

    @Override
    public String getName();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public HotSpotResolvedObjectTypeImpl getDeclaringClass();

    public Constant getMetaspaceMethodConstant();

    long getMetaspaceMethod();

    @Override
    public long getMetadataHandle();

    @Override
    public Constant getEncoding();

    public int getAllModifiers();

    @Override
    public int getModifiers();

    @Override
    public boolean canBeStaticallyBound();

    @Override
    public byte[] getCode();

    @Override
    public int getCodeSize();

    @Override
    public ExceptionHandler[] getExceptionHandlers();

    @Override
    public boolean isCallerSensitive();

    @Override
    public boolean isForceInline();

    @Override
    public boolean hasReservedStackAccess();

    @Override
    public void setNotInlinableOrCompilable();

    @Override
    public boolean ignoredBySecurityStackWalk();

    @Override
    public boolean isClassInitializer();

    @Override
    public boolean isConstructor();

    @Override
    public int getMaxLocals();

    @Override
    public int getMaxStackSize();

    @Override
    public StackTraceElement asStackTraceElement(int bci);

    @Override
    public ResolvedJavaMethod uniqueConcreteMethod(HotSpotResolvedObjectType receiver);

    @Override
    public HotSpotSignature getSignature();

    @Override
    public boolean hasCompiledCode();

    @Override
    public boolean hasCompiledCodeAtLevel(int level);

    @Override
    public ProfilingInfo getProfilingInfo(boolean includeNormal, boolean includeOSR);

    @Override
    public void reprofile();

    @Override
    public ConstantPool getConstantPool();

    @Override
    public Parameter[] getParameters();

    @Override
    public Annotation[][] getParameterAnnotations();

    @Override
    public Annotation[] getAnnotations();

    @Override
    public Annotation[] getDeclaredAnnotations();

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Override
    public boolean isBridge();

    @Override
    public boolean isSynthetic();

    @Override
    public boolean isVarArgs();

    @Override
    public boolean isDefault();

    @Override
    public Type[] getGenericParameterTypes();

    @Override
    public boolean canBeInlined();

    @Override
    public boolean hasNeverInlineDirective();

    @Override
    public boolean shouldBeInlined();

    @Override
    public LineNumberTable getLineNumberTable();

    @Override
    public LocalVariableTable getLocalVariableTable();

    @Override
    public int vtableEntryOffset(ResolvedJavaType resolved);

    @Override
    public boolean isInVirtualMethodTable(ResolvedJavaType resolved);

    @Override
    public SpeculationLog getSpeculationLog();

    @Override
    public int intrinsicId();

    @Override
    public boolean isIntrinsicCandidate();

    @Override
    public int allocateCompileId(int entryBCI);

    @Override
    public boolean hasCodeAtLevel(int entryBCI, int level);

    @Override
    public int methodIdnum();
}
