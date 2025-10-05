/*
    @Positive
 * Copyright (c) 2011, 2019, Oracle and/or its affiliates. All rights reserved.
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
import static jdk.vm.ci.hotspot.CompilerToVM.compilerToVM;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.runtime;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotModifiers.BRIDGE;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotModifiers.SYNTHETIC;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotModifiers.VARARGS;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotModifiers.jvmMethodModifiers;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotVMConfig.config;
    @Positive
import static jdk.vm.ci.hotspot.UnsafeAccess.UNSAFE;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.reflect.Executable;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.lang.reflect.Type;
    @Positive
import jdk.vm.ci.common.JVMCIError;
    @Positive
import jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.Option;
    @Positive
import jdk.vm.ci.meta.Constant;
    @Positive
import jdk.vm.ci.meta.ConstantPool;
    @Positive
import jdk.vm.ci.meta.DefaultProfilingInfo;
    @Positive
import jdk.vm.ci.meta.ExceptionHandler;
    @Positive
import jdk.vm.ci.meta.JavaMethod;
    @Positive
import jdk.vm.ci.meta.JavaType;
    @Positive
import jdk.vm.ci.meta.LineNumberTable;
    @Positive
import jdk.vm.ci.meta.Local;
    @Positive
import jdk.vm.ci.meta.LocalVariableTable;
    @Positive
import jdk.vm.ci.meta.ProfilingInfo;
    @Positive
import jdk.vm.ci.meta.ResolvedJavaMethod;
    @Positive
import jdk.vm.ci.meta.ResolvedJavaType;
    @Positive
import jdk.vm.ci.meta.SpeculationLog;
    @Positive
import jdk.vm.ci.meta.TriState;

    @Positive
final class HotSpotResolvedJavaMethodImpl extends HotSpotMethod implements HotSpotResolvedJavaMethod, MetaspaceHandleObject {

    @Positive
    @Override
    @Positive
    public String getName();

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
    public int hashCode();

    @Positive
    @Override
    @Positive
    public HotSpotResolvedObjectTypeImpl getDeclaringClass();

    @Positive
    public Constant getMetaspaceMethodConstant();

    @Positive
    long getMetaspaceMethod();

    @Positive
    @Override
    @Positive
    public long getMetadataHandle();

    @Positive
    @Override
    @Positive
    public Constant getEncoding();

    @Positive
    public int getAllModifiers();

    @Positive
    @Override
    @Positive
    public int getModifiers();

    @Positive
    @Override
    @Positive
    public boolean canBeStaticallyBound();

    @Positive
    @Override
    @Positive
    public byte[] getCode();

    @Positive
    @Override
    @Positive
    public int getCodeSize();

    @Positive
    @Override
    @Positive
    public ExceptionHandler[] getExceptionHandlers();

    @Positive
    @Override
    @Positive
    public boolean isCallerSensitive();

    @Positive
    @Override
    @Positive
    public boolean isForceInline();

    @Positive
    @Override
    @Positive
    public boolean hasReservedStackAccess();

    @Positive
    @Override
    @Positive
    public void setNotInlinableOrCompilable();

    @Positive
    @Override
    @Positive
    public boolean ignoredBySecurityStackWalk();

    @Positive
    @Override
    @Positive
    public boolean isClassInitializer();

    @Positive
    @Override
    @Positive
    public boolean isConstructor();

    @Positive
    @Override
    @Positive
    public int getMaxLocals();

    @Positive
    @Override
    @Positive
    public int getMaxStackSize();

    @Positive
    @Override
    @Positive
    public StackTraceElement asStackTraceElement(int bci);

    @Positive
    @Override
    @Positive
    public ResolvedJavaMethod uniqueConcreteMethod(HotSpotResolvedObjectType receiver);

    @Positive
    @Override
    @Positive
    public HotSpotSignature getSignature();

    @Positive
    @Override
    @Positive
    public boolean hasCompiledCode();

    @Positive
    @Override
    @Positive
    public boolean hasCompiledCodeAtLevel(int level);

    @Positive
    @Override
    @Positive
    public ProfilingInfo getProfilingInfo(boolean includeNormal, boolean includeOSR);

    @Positive
    @Override
    @Positive
    public void reprofile();

    @Positive
    @Override
    @Positive
    public ConstantPool getConstantPool();

    @Positive
    @Override
    @Positive
    public Parameter[] getParameters();

    @Positive
    @Override
    @Positive
    public Annotation[][] getParameterAnnotations();

    @Positive
    @Override
    @Positive
    public Annotation[] getAnnotations();

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public boolean isBridge();

    @Positive
    @Override
    @Positive
    public boolean isSynthetic();

    @Positive
    @Override
    @Positive
    public boolean isVarArgs();

    @Positive
    @Override
    @Positive
    public boolean isDefault();

    @Positive
    @Override
    @Positive
    public Type[] getGenericParameterTypes();

    @Positive
    @Override
    @Positive
    public boolean canBeInlined();

    @Positive
    @Override
    @Positive
    public boolean hasNeverInlineDirective();

    @Positive
    @Override
    @Positive
    public boolean shouldBeInlined();

    @Positive
    @Override
    @Positive
    public LineNumberTable getLineNumberTable();

    @Positive
    @Override
    @Positive
    public LocalVariableTable getLocalVariableTable();

    @Positive
    @Override
    @Positive
    public int vtableEntryOffset(ResolvedJavaType resolved);

    @Positive
    @Override
    @Positive
    public boolean isInVirtualMethodTable(ResolvedJavaType resolved);

    @Positive
    @Override
    @Positive
    public SpeculationLog getSpeculationLog();

    @Positive
    @Override
    @Positive
    public int intrinsicId();

    @Positive
    @Override
    @Positive
    public boolean isIntrinsicCandidate();

    @Positive
    @Override
    @Positive
    public int allocateCompileId(int entryBCI);

    @Positive
    @Override
    @Positive
    public boolean hasCodeAtLevel(int entryBCI, int level);

    @Positive
    @Override
    @Positive
    public int methodIdnum();
    @Positive
}

// CFWR semantic augmentation - variant 1
