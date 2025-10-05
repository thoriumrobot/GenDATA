/*
 * Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.reflect;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.annotation.*;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.stream.Stream;
import java.util.stream.Collectors;
import jdk.internal.access.SharedSecrets;
import sun.reflect.annotation.AnnotationParser;
import sun.reflect.annotation.AnnotationSupport;
import sun.reflect.annotation.TypeAnnotationParser;
import sun.reflect.annotation.TypeAnnotation;
import sun.reflect.generics.reflectiveObjects.ParameterizedTypeImpl;
import sun.reflect.generics.repository.ConstructorRepository;

@AnnotatedFor({ "nullness" })
public abstract sealed class Executable extends AccessibleObject implements Member, GenericDeclaration permits Constructor, Method {

    abstract byte[] getAnnotationBytes();

    abstract boolean hasGenericInformation();

    abstract ConstructorRepository getGenericInfo();

    boolean equalParamTypes(Class<?>[] params1, Class<?>[] params2);

    Annotation[][] parseParameterAnnotations(byte[] parameterAnnotations);

    void printModifiersIfNonzero(StringBuilder sb, int mask, boolean isDefault);

    String sharedToString(int modifierMask, boolean isDefault, Class<?>[] parameterTypes, Class<?>[] exceptionTypes);

    abstract void specificToStringHeader(StringBuilder sb);

    static String typeVarBounds(TypeVariable<?> typeVar);

    String sharedToGenericString(int modifierMask, boolean isDefault);

    abstract void specificToGenericStringHeader(StringBuilder sb);

    public abstract Class<?> getDeclaringClass();

    public abstract String getName();

    public abstract int getModifiers();

    public abstract TypeVariable<?>[] getTypeParameters();

    abstract Class<?>[] getSharedParameterTypes();

    abstract Class<?>[] getSharedExceptionTypes();

    public abstract Class<?>[] getParameterTypes();

    public int getParameterCount();

    public Type[] getGenericParameterTypes();

    Type[] getAllGenericParameterTypes();

    public Parameter[] getParameters();

    boolean hasRealParameterData();

    native byte[] getTypeAnnotationBytes0();

    byte[] getTypeAnnotationBytes();

    public abstract Class<?>[] getExceptionTypes();

    public Type[] getGenericExceptionTypes();

    public abstract String toGenericString();

    public boolean isVarArgs();

    public boolean isSynthetic();

    public abstract Annotation[][] getParameterAnnotations();

    Annotation[][] sharedGetParameterAnnotations(Class<?>[] parameterTypes, byte[] parameterAnnotations);

    abstract boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);

    @Override
    @Nullable
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Override
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @Override
    public Annotation[] getDeclaredAnnotations();

    public abstract AnnotatedType getAnnotatedReturnType();

    AnnotatedType getAnnotatedReturnType0(Type returnType);

    @Nullable
    public AnnotatedType getAnnotatedReceiverType();

    Type parameterize(Class<?> c);

    public AnnotatedType[] getAnnotatedParameterTypes();

    public AnnotatedType[] getAnnotatedExceptionTypes();
}
