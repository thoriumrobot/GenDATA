/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.security.AccessController;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.reflect.ReflectionFactory;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class AccessibleObject implements AnnotatedElement {

    @Positive
    static void checkPermission();

    @Positive
    @CallerSensitive
    @Positive
    public static void setAccessible(AccessibleObject[] array, boolean flag);

    @Positive
    @CallerSensitive
    @Positive
    public void setAccessible(boolean flag);

    @Positive
    boolean setAccessible0(boolean flag);

    @Positive
    @CallerSensitive
    @Positive
    public final boolean trySetAccessible();

    @Positive
    void checkCanSetAccessible(Class<?> caller);

    @Positive
    final void checkCanSetAccessible(Class<?> caller, Class<?> declaringClass);

    @Positive
    String toShortString();

    @Positive
    @Deprecated()
    @Positive
    public boolean isAccessible();

    @Positive
    @CallerSensitive
    @Positive
    @CFComment("Sometimes null is forbidden; other times, it is required")
    @Positive
    public final boolean canAccess(Object obj);

    @Positive
    @Deprecated()
    @Positive
    protected AccessibleObject() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public boolean isAnnotationPresent(Class<? extends Annotation> annotationClass);

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getAnnotations();

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
    public Annotation[] getDeclaredAnnotations();

    @Positive
    private static class Cache {

    @Positive
        boolean isCacheFor(Class<?> caller, Class<?> refc);

    @Positive
        static Object protectedMemberCallerCache(Class<?> caller, Class<?> refc);
    @Positive
    }

    @Positive
    final void checkAccess(Class<?> caller, Class<?> memberClass, Class<?> targetClass, int modifiers) throws IllegalAccessException;

    @Positive
    final boolean verifyAccess(Class<?> caller, Class<?> memberClass, Class<?> targetClass, int modifiers);

    @Positive
    AccessibleObject getRoot();
    @Positive
}

// CFWR semantic augmentation - variant 0
