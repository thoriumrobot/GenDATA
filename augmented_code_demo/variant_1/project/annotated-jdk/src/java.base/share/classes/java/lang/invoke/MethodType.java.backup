/*
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.invoke;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.constant.ClassDesc;
import java.lang.constant.Constable;
import java.lang.constant.MethodTypeDesc;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.StringJoiner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Stream;
import jdk.internal.vm.annotation.Stable;
import sun.invoke.util.BytecodeDescriptor;
import sun.invoke.util.VerifyType;
import sun.invoke.util.Wrapper;
import sun.security.util.SecurityConstants;
import static java.lang.invoke.MethodHandleStatics.UNSAFE;
import static java.lang.invoke.MethodHandleStatics.newIllegalArgumentException;
import static java.lang.invoke.MethodType.fromDescriptor;

public final class MethodType implements Constable, TypeDescriptor.OfMethod<Class<?>, MethodType>, java.io.Serializable {

    MethodTypeForm form();

    Class<?> rtype();

    Class<?>[] ptypes();

    void setForm(MethodTypeForm f);

    static void checkSlotCount(int count);

    public static MethodType methodType(Class<?> rtype, Class<?>[] ptypes);

    public static MethodType methodType(Class<?> rtype, List<Class<?>> ptypes);

    public static MethodType methodType(Class<?> rtype, Class<?> ptype0, Class<?>... ptypes);

    public static MethodType methodType(Class<?> rtype);

    public static MethodType methodType(Class<?> rtype, Class<?> ptype0);

    public static MethodType methodType(Class<?> rtype, MethodType ptypes);

    static MethodType makeImpl(Class<?> rtype, Class<?>[] ptypes, boolean trusted);

    public static MethodType genericMethodType(int objectArgCount, boolean finalArray);

    public static MethodType genericMethodType(int objectArgCount);

    public MethodType changeParameterType(int num, Class<?> nptype);

    public MethodType insertParameterTypes(int num, Class<?>... ptypesToInsert);

    public MethodType appendParameterTypes(Class<?>... ptypesToInsert);

    public MethodType insertParameterTypes(int num, List<Class<?>> ptypesToInsert);

    public MethodType appendParameterTypes(List<Class<?>> ptypesToInsert);

    MethodType replaceParameterTypes(int start, int end, Class<?>... ptypesToInsert);

    MethodType asSpreaderType(Class<?> arrayType, int pos, int arrayLength);

    Class<?> leadingReferenceParameter();

    MethodType asCollectorType(Class<?> arrayType, int pos, int arrayLength);

    public MethodType dropParameterTypes(int start, int end);

    public MethodType changeReturnType(Class<?> nrtype);

    public boolean hasPrimitives();

    public boolean hasWrappers();

    public MethodType erase();

    MethodType basicType();

    MethodType invokerType();

    public MethodType generic();

    boolean isGeneric();

    public MethodType wrap();

    public MethodType unwrap();

    public Class<?> parameterType(int num);

    public int parameterCount();

    public Class<?> returnType();

    public List<Class<?>> parameterList();

    public Class<?> lastParameterType();

    public Class<?>[] parameterArray();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object x);

    @Override
    public int hashCode();

    @Override
    public String toString();

    boolean effectivelyIdenticalParameters(int skipPos, List<Class<?>> fullList);

    boolean isViewableAs(MethodType newType, boolean keepInterfaces);

    boolean isConvertibleTo(MethodType newType);

    boolean explicitCastEquivalentToAsType(MethodType newType);

    static boolean canConvert(Class<?> src, Class<?> dst);

    int parameterSlotCount();

    Invokers invokers();

    public static MethodType fromMethodDescriptorString(String descriptor, ClassLoader loader) throws IllegalArgumentException, TypeNotPresentException;

    static MethodType fromDescriptor(String descriptor, ClassLoader loader) throws IllegalArgumentException, TypeNotPresentException;

    public String toMethodDescriptorString();

    @Override
    public String descriptorString();

    static String toFieldDescriptorString(Class<?> cls);

    @Override
    public Optional<MethodTypeDesc> describeConstable();

    private static class OffsetHolder {
    }

    private static class ConcurrentWeakInternSet<T> {

        public ConcurrentWeakInternSet() {
        }

        public T get(T elem);

        public T add(T elem);

        private static class WeakEntry<T> extends WeakReference<T> {

            public final int hashcode;

            public WeakEntry(T key, ReferenceQueue<T> queue) {
            }

            @Override
            public boolean equals(Object obj);

            @Override
            public int hashCode();
        }
    }
}
