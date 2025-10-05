/*
    @Positive
 * Copyright (c) 2010, 2013, Oracle and/or its affiliates. All rights reserved.
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
package jdk.dynalink.beans;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.MethodType;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.StringTokenizer;
    @Positive
import jdk.dynalink.CallSiteDescriptor;
    @Positive
import jdk.dynalink.linker.LinkerServices;
    @Positive
import jdk.dynalink.linker.support.Guards;
    @Positive
import jdk.dynalink.linker.support.Lookup;

    @Positive
abstract class SingleDynamicMethod extends DynamicMethod {

    @Positive
    abstract boolean isVarArgs();

    @Positive
    abstract MethodType getMethodType();

    @Positive
    abstract MethodHandle getTarget(CallSiteDescriptor desc);

    @Positive
    @Override
    @Positive
    MethodHandle getInvocation(final CallSiteDescriptor callSiteDescriptor, final LinkerServices linkerServices);

    @Positive
    @Override
    @Positive
    SingleDynamicMethod getMethodForExactParamTypes(final String paramTypes);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    boolean contains(final SingleDynamicMethod method);

    @Positive
    static String getMethodNameWithSignature(final MethodType type, final String methodName, final boolean withReturnType);

    @Positive
    static MethodHandle getInvocation(final MethodHandle target, final MethodType callSiteType, final LinkerServices linkerServices);

    @Positive
    static MethodHandle collectArguments(final MethodHandle target, final int parameterCount);
    @Positive
}

// CFWR semantic augmentation - variant 0
