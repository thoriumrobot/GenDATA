/*
 * Copyright (c) 1996, 2015, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.ref.Reference;
import java.lang.reflect.Method;
import java.lang.reflect.Constructor;
import java.util.Map.Entry;
import com.sun.beans.introspect.PropertyInfo;
import sun.reflect.misc.ReflectUtil;

@AnnotatedFor({ "nullness" })
public class PropertyDescriptor extends FeatureDescriptor {

    @SideEffectFree
    public PropertyDescriptor(String propertyName, Class<?> beanClass) throws IntrospectionException {
    }

    @SideEffectFree
    public PropertyDescriptor(String propertyName, Class<?> beanClass, @Nullable String readMethodName, @Nullable String writeMethodName) throws IntrospectionException {
    }

    @SideEffectFree
    public PropertyDescriptor(String propertyName, @Nullable Method readMethod, @Nullable Method writeMethod) throws IntrospectionException {
    }

    @Pure
    @Nullable
    public synchronized Class<?> getPropertyType();

    @Pure
    @Nullable
    public synchronized Method getReadMethod();

    public synchronized void setReadMethod(@Nullable Method readMethod) throws IntrospectionException;

    @Pure
    @Nullable
    public synchronized Method getWriteMethod();

    public synchronized void setWriteMethod(@Nullable Method writeMethod) throws IntrospectionException;

    void setClass0(Class<?> clz);

    @Pure
    public boolean isBound();

    public void setBound(boolean bound);

    @Pure
    public boolean isConstrained();

    public void setConstrained(boolean constrained);

    public void setPropertyEditorClass(@Nullable Class<?> propertyEditorClass);

    @Pure
    @Nullable
    public Class<?> getPropertyEditorClass();

    @SideEffectFree
    @SuppressWarnings("deprecation")
    @Nullable
    public PropertyEditor createPropertyEditor(@Nullable Object bean);

    @Pure
    public boolean equals(@Nullable Object obj);

    boolean compareMethods(@Nullable Method a, @Nullable Method b);

    void updateGenericsFor(Class<?> type);

    @Pure
    public int hashCode();

    String getBaseName();

    void appendTo(StringBuilder sb);

    boolean isAssignable(@Nullable Method m1, @Nullable Method m2);
}
