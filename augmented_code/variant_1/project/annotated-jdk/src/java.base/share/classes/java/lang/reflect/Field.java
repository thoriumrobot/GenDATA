/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
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
import org.checkerframework.framework.qual.CFComment;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.FieldAccessor;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import sun.reflect.generics.repository.FieldRepository;
    @Positive
import sun.reflect.generics.factory.CoreReflectionFactory;
    @Positive
import sun.reflect.generics.factory.GenericsFactory;
    @Positive
import sun.reflect.generics.scope.ClassScope;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import sun.reflect.annotation.AnnotationParser;
    @Positive
import sun.reflect.annotation.AnnotationSupport;
    @Positive
import sun.reflect.annotation.TypeAnnotation;
    @Positive
import sun.reflect.annotation.TypeAnnotationParser;

    @Positive
@CFComment({ "In general, the field value 'get' methods should take a top-qualified 'obj' parameter ", "and have a top-qualified return type; the field value 'set' methods should take a ", "top-qualified 'obj' parameter and a bottom-qualified 'value' parameter.", "nullness: the 'obj' parameter in 'get' or 'set' methods is @NonNull, because instance fields ", "require a receiver. Static field accesses need to suppress the errors.", "initialization: using fully-initialized types should make the typical use case easier.", "lock: require @GuardSatisfied to ensure type system soundness." })
    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
public final class Field extends AccessibleObject implements Member {

    @Positive
    Field copy();

    @Positive
    @Override
    @Positive
    @CallerSensitive
    @Positive
    public void setAccessible(boolean flag);

    @Positive
    @Override
    @Positive
    void checkCanSetAccessible(Class<?> caller);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Class<?> getDeclaringClass(@GuardSatisfied Field this);

    @Positive
    @SideEffectFree
    @Positive
    @Interned
    @Positive
    public String getName(@GuardSatisfied Field this);

    @Positive
    @Pure
    @Positive
    public int getModifiers(@GuardSatisfied Field this);

    @Positive
    @Pure
    @Positive
    public boolean isEnumConstant(@GuardSatisfied Field this);

    @Positive
    @Pure
    @Positive
    public boolean isSynthetic(@GuardSatisfied Field this);

    @Positive
    @SideEffectFree
    @Positive
    public Class<?> getType(@GuardSatisfied Field this);

    @Positive
    @SideEffectFree
    @Positive
    public Type getGenericType(@GuardSatisfied Field this);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied Field this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied Field this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Field this);

    @Positive
    @Override
    @Positive
    String toShortString();

    @Positive
    @SideEffectFree
    @Positive
    public String toGenericString(@GuardSatisfied Field this);

    @Positive
    @SideEffectFree
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    @Nullable
    @Positive
    public Object get(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public boolean getBoolean(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public byte getByte(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public char getChar(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public short getShort(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public int getInt(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public long getLong(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public float getFloat(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public double getDouble(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void set(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, @GuardSatisfied @Interned Object value) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setBoolean(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, boolean z) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setByte(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, byte b) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setChar(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, char c) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setShort(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, short s) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setInt(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, int i) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setLong(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, long l) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setFloat(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, float f) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public void setDouble(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, double d) throws IllegalArgumentException, IllegalAccessException;

    @Positive
    @Override
    @Positive
    Field getRoot();

    @Positive
    boolean isTrustedFinal();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <T extends Annotation> T getAnnotation(@GuardSatisfied Field this, Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations(@GuardSatisfied Field this);

    @Positive
    public AnnotatedType getAnnotatedType();
    @Positive
}
