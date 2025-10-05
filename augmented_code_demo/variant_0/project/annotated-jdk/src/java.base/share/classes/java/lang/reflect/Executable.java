/*
    @Positive
 * Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.reflect;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.annotation.*;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.Collectors;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.reflect.annotation.AnnotationParser;
    @Positive
import sun.reflect.annotation.AnnotationSupport;
    @Positive
import sun.reflect.annotation.TypeAnnotationParser;
    @Positive
import sun.reflect.annotation.TypeAnnotation;
    @Positive
import sun.reflect.generics.reflectiveObjects.ParameterizedTypeImpl;
    @Positive
import sun.reflect.generics.repository.ConstructorRepository;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public abstract sealed class Executable extends AccessibleObject implements Member, GenericDeclaration permits Constructor, Method {

    @Positive
    abstract byte[] getAnnotationBytes();

    @Positive
    abstract boolean hasGenericInformation();

    @Positive
    abstract ConstructorRepository getGenericInfo();

    @Positive
    boolean equalParamTypes(Class<?>[] params1, Class<?>[] params2);

    @Positive
    Annotation[][] parseParameterAnnotations(byte[] parameterAnnotations);

    @Positive
    void printModifiersIfNonzero(StringBuilder sb, int mask, boolean isDefault);

    @Positive
    String sharedToString(int modifierMask, boolean isDefault, Class<?>[] parameterTypes, Class<?>[] exceptionTypes);

    @Positive
    abstract void specificToStringHeader(StringBuilder sb);

    @Positive
    static String typeVarBounds(TypeVariable<?> typeVar);

    @Positive
    String sharedToGenericString(int modifierMask, boolean isDefault);

    @Positive
    abstract void specificToGenericStringHeader(StringBuilder sb);

    @Positive
    public abstract Class<?> getDeclaringClass();

    @Positive
    public abstract String getName();

    @Positive
    public abstract int getModifiers();

    @Positive
    public abstract TypeVariable<?>[] getTypeParameters();

    @Positive
    abstract Class<?>[] getSharedParameterTypes();

    @Positive
    abstract Class<?>[] getSharedExceptionTypes();

    @Positive
    public abstract Class<?>[] getParameterTypes();

    @Positive
    public int getParameterCount();

    @Positive
    public Type[] getGenericParameterTypes();

    @Positive
    Type[] getAllGenericParameterTypes();

    @Positive
    public Parameter[] getParameters();

    @Positive
    boolean hasRealParameterData();

    @Positive
    native byte[] getTypeAnnotationBytes0();

    @Positive
    byte[] getTypeAnnotationBytes();

    @Positive
    public abstract Class<?>[] getExceptionTypes();

    @Positive
    public Type[] getGenericExceptionTypes();

    @Positive
    public abstract String toGenericString();

    @Positive
    public boolean isVarArgs();

    @Positive
    public boolean isSynthetic();

    @Positive
    public abstract Annotation[][] getParameterAnnotations();

    @Positive
    Annotation[][] sharedGetParameterAnnotations(Class<?>[] parameterTypes, byte[] parameterAnnotations);

    @Positive
    abstract boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    public abstract AnnotatedType getAnnotatedReturnType();

    @Positive
    AnnotatedType getAnnotatedReturnType0(Type returnType);

    @Positive
    @Nullable
    @Positive
    public AnnotatedType getAnnotatedReceiverType();

    @Positive
    Type parameterize(Class<?> c);

    @Positive
    public AnnotatedType[] getAnnotatedParameterTypes();

    @Positive
    public AnnotatedType[] getAnnotatedExceptionTypes();
    @Positive
}

// CFWR semantic augmentation - variant 0
