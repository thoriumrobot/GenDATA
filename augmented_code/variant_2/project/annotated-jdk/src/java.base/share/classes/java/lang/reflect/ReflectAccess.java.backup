/*
 * Copyright (c) 2001, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.reflect.MethodAccessor;
import jdk.internal.reflect.ConstructorAccessor;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
class ReflectAccess implements jdk.internal.access.JavaLangReflectAccess {

    public <T> Constructor<T> newConstructor(Class<T> declaringClass, Class<?>[] parameterTypes, Class<?>[] checkedExceptions, int modifiers, int slot, String signature, byte[] annotations, byte[] parameterAnnotations);

    public MethodAccessor getMethodAccessor(Method m);

    public void setMethodAccessor(Method m, MethodAccessor accessor);

    public ConstructorAccessor getConstructorAccessor(Constructor<?> c);

    public void setConstructorAccessor(Constructor<?> c, ConstructorAccessor accessor);

    public int getConstructorSlot(Constructor<?> c);

    public String getConstructorSignature(Constructor<?> c);

    public byte[] getConstructorAnnotations(Constructor<?> c);

    public byte[] getConstructorParameterAnnotations(Constructor<?> c);

    public byte[] getExecutableTypeAnnotationBytes(Executable ex);

    public Class<?>[] getExecutableSharedParameterTypes(Executable ex);

    public Method copyMethod(Method arg);

    public Method leafCopyMethod(Method arg);

    public Field copyField(Field arg);

    public <T> Constructor<T> copyConstructor(Constructor<T> arg);

    @SuppressWarnings("unchecked")
    public <T extends AccessibleObject> T getRoot(T obj);

    public boolean isTrustedFinalField(Field f);

    public <T> T newInstance(Constructor<T> ctor, Object[] args, Class<?> caller) throws IllegalAccessException, InstantiationException, InvocationTargetException;
}
