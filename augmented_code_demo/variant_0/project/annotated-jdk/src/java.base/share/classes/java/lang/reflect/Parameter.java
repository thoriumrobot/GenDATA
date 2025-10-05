/*
    @Positive
 * Copyright (c) 2013, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.annotation.*;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import sun.reflect.annotation.AnnotationSupport;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public final class Parameter implements AnnotatedElement {

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    @Override
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    public boolean isNamePresent();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public Executable getDeclaringExecutable();

    @Positive
    public int getModifiers();

    @Positive
    public String getName();

    @Positive
    String getRealName();

    @Positive
    public Type getParameterizedType();

    @Positive
    public Class<?> getType();

    @Positive
    public AnnotatedType getAnnotatedType();

    @Positive
    public boolean isImplicit();

    @Positive
    public boolean isSynthetic();

    @Positive
    public boolean isVarArgs();

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
    @Override
    @Positive
    @Nullable
    @Positive
    public <T extends Annotation> T getDeclaredAnnotation(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T[] getDeclaredAnnotationsByType(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getAnnotations();
    @Positive
}

// CFWR semantic augmentation - variant 0
