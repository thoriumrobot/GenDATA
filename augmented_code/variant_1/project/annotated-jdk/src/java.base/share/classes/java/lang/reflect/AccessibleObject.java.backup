/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.framework.qual.CFComment;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandle;
import java.lang.ref.WeakReference;
import java.security.AccessController;
import jdk.internal.access.SharedSecrets;
import jdk.internal.misc.VM;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import jdk.internal.reflect.ReflectionFactory;
import sun.security.action.GetPropertyAction;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "nullness" })
public class AccessibleObject implements AnnotatedElement {

    static void checkPermission();

    @CallerSensitive
    public static void setAccessible(AccessibleObject[] array, boolean flag);

    @CallerSensitive
    public void setAccessible(boolean flag);

    boolean setAccessible0(boolean flag);

    @CallerSensitive
    public final boolean trySetAccessible();

    void checkCanSetAccessible(Class<?> caller);

    final void checkCanSetAccessible(Class<?> caller, Class<?> declaringClass);

    String toShortString();

    @Deprecated()
    public boolean isAccessible();

    @CallerSensitive
    @CFComment("Sometimes null is forbidden; other times, it is required")
    public final boolean canAccess(Object obj);

    @Deprecated()
    protected AccessibleObject() {
    }

    @Override
    @Nullable
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Override
    public boolean isAnnotationPresent(Class<? extends Annotation> annotationClass);

    @Override
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @Override
    public Annotation[] getAnnotations();

    @Override
    @Nullable
    public <T extends Annotation> T getDeclaredAnnotation(Class<T> annotationClass);

    @Override
    public <T extends Annotation> T[] getDeclaredAnnotationsByType(Class<T> annotationClass);

    @Override
    public Annotation[] getDeclaredAnnotations();

    private static class Cache {

        boolean isCacheFor(Class<?> caller, Class<?> refc);

        static Object protectedMemberCallerCache(Class<?> caller, Class<?> refc);
    }

    final void checkAccess(Class<?> caller, Class<?> memberClass, Class<?> targetClass, int modifiers) throws IllegalAccessException;

    final boolean verifyAccess(Class<?> caller, Class<?> memberClass, Class<?> targetClass, int modifiers);

    AccessibleObject getRoot();
}
