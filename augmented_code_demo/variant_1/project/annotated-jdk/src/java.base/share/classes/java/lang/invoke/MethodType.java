/*
    @Positive
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.invoke;

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
import java.lang.constant.ClassDesc;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.MethodTypeDesc;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.invoke.util.BytecodeDescriptor;
    @Positive
import sun.invoke.util.VerifyType;
    @Positive
import sun.invoke.util.Wrapper;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import static java.lang.invoke.MethodHandleStatics.UNSAFE;
    @Positive
import static java.lang.invoke.MethodHandleStatics.newIllegalArgumentException;
    @Positive
import static java.lang.invoke.MethodType.fromDescriptor;

    @Positive
public final class MethodType implements Constable, TypeDescriptor.OfMethod<Class<?>, MethodType>, java.io.Serializable {

    @Positive
    MethodTypeForm form();

    @Positive
    Class<?> rtype();

    @Positive
    Class<?>[] ptypes();

    @Positive
    void setForm(MethodTypeForm f);

    @Positive
    static void checkSlotCount(int count);

    @Positive
    public static MethodType methodType(Class<?> rtype, Class<?>[] ptypes);

    @Positive
    public static MethodType methodType(Class<?> rtype, List<Class<?>> ptypes);

    @Positive
    public static MethodType methodType(Class<?> rtype, Class<?> ptype0, Class<?>... ptypes);

    @Positive
    public static MethodType methodType(Class<?> rtype);

    @Positive
    public static MethodType methodType(Class<?> rtype, Class<?> ptype0);

    @Positive
    public static MethodType methodType(Class<?> rtype, MethodType ptypes);

    @Positive
    static MethodType makeImpl(Class<?> rtype, Class<?>[] ptypes, boolean trusted);

    @Positive
    public static MethodType genericMethodType(int objectArgCount, boolean finalArray);

    @Positive
    public static MethodType genericMethodType(int objectArgCount);

    @Positive
    public MethodType changeParameterType(int num, Class<?> nptype);

    @Positive
    public MethodType insertParameterTypes(int num, Class<?>... ptypesToInsert);

    @Positive
    public MethodType appendParameterTypes(Class<?>... ptypesToInsert);

    @Positive
    public MethodType insertParameterTypes(int num, List<Class<?>> ptypesToInsert);

    @Positive
    public MethodType appendParameterTypes(List<Class<?>> ptypesToInsert);

    @Positive
    MethodType replaceParameterTypes(int start, int end, Class<?>... ptypesToInsert);

    @Positive
    MethodType asSpreaderType(Class<?> arrayType, int pos, int arrayLength);

    @Positive
    Class<?> leadingReferenceParameter();

    @Positive
    MethodType asCollectorType(Class<?> arrayType, int pos, int arrayLength);

    @Positive
    public MethodType dropParameterTypes(int start, int end);

    @Positive
    public MethodType changeReturnType(Class<?> nrtype);

    @Positive
    public boolean hasPrimitives();

    @Positive
    public boolean hasWrappers();

    @Positive
    public MethodType erase();

    @Positive
    MethodType basicType();

    @Positive
    MethodType invokerType();

    @Positive
    public MethodType generic();

    @Positive
    boolean isGeneric();

    @Positive
    public MethodType wrap();

    @Positive
    public MethodType unwrap();

    @Positive
    public Class<?> parameterType(int num);

    @Positive
    public int parameterCount();

    @Positive
    public Class<?> returnType();

    @Positive
    public List<Class<?>> parameterList();

    @Positive
    public Class<?> lastParameterType();

    @Positive
    public Class<?>[] parameterArray();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object x);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    boolean effectivelyIdenticalParameters(int skipPos, List<Class<?>> fullList);

    @Positive
    boolean isViewableAs(MethodType newType, boolean keepInterfaces);

    @Positive
    boolean isConvertibleTo(MethodType newType);

    @Positive
    boolean explicitCastEquivalentToAsType(MethodType newType);

    @Positive
    static boolean canConvert(Class<?> src, Class<?> dst);

    @Positive
    int parameterSlotCount();

    @Positive
    Invokers invokers();

    @Positive
    public static MethodType fromMethodDescriptorString(String descriptor, ClassLoader loader) throws IllegalArgumentException, TypeNotPresentException;

    @Positive
    static MethodType fromDescriptor(String descriptor, ClassLoader loader) throws IllegalArgumentException, TypeNotPresentException;

    @Positive
    public String toMethodDescriptorString();

    @Positive
    @Override
    @Positive
    public String descriptorString();

    @Positive
    static String toFieldDescriptorString(Class<?> cls);

    @Positive
    @Override
    @Positive
    public Optional<MethodTypeDesc> describeConstable();

    @Positive
    private static class OffsetHolder {
    @Positive
    }

    @Positive
    private static class ConcurrentWeakInternSet<T> {

    @Positive
        public ConcurrentWeakInternSet() {
    @Positive
        }

    @Positive
        public T get(T elem);

    @Positive
        public T add(T elem);

    @Positive
        private static class WeakEntry<T> extends WeakReference<T> {

    @Positive
            public final int hashcode;

    @Positive
            public WeakEntry(T key, ReferenceQueue<T> queue) {
    @Positive
            }

    @Positive
            @Override
    @Positive
            public boolean equals(Object obj);

    @Positive
            @Override
    @Positive
            public int hashCode();
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
