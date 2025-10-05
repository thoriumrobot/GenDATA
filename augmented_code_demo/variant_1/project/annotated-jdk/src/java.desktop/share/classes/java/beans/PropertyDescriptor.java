/*
    @Positive
 * Copyright (c) 1996, 2015, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.util.Map.Entry;
    @Positive
import com.sun.beans.introspect.PropertyInfo;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class PropertyDescriptor extends FeatureDescriptor {

    @Positive
    @SideEffectFree
    @Positive
    public PropertyDescriptor(String propertyName, Class<?> beanClass) throws IntrospectionException {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public PropertyDescriptor(String propertyName, Class<?> beanClass, @Nullable String readMethodName, @Nullable String writeMethodName) throws IntrospectionException {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public PropertyDescriptor(String propertyName, @Nullable Method readMethod, @Nullable Method writeMethod) throws IntrospectionException {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public synchronized Class<?> getPropertyType();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public synchronized Method getReadMethod();

    @Positive
    public synchronized void setReadMethod(@Nullable Method readMethod) throws IntrospectionException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public synchronized Method getWriteMethod();

    @Positive
    public synchronized void setWriteMethod(@Nullable Method writeMethod) throws IntrospectionException;

    @Positive
    void setClass0(Class<?> clz);

    @Positive
    @Pure
    @Positive
    public boolean isBound();

    @Positive
    public void setBound(boolean bound);

    @Positive
    @Pure
    @Positive
    public boolean isConstrained();

    @Positive
    public void setConstrained(boolean constrained);

    @Positive
    public void setPropertyEditorClass(@Nullable Class<?> propertyEditorClass);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Class<?> getPropertyEditorClass();

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    @Nullable
    @Positive
    public PropertyEditor createPropertyEditor(@Nullable Object bean);

    @Positive
    @Pure
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    boolean compareMethods(@Nullable Method a, @Nullable Method b);

    @Positive
    void updateGenericsFor(Class<?> type);

    @Positive
    @Pure
    @Positive
    public int hashCode();

    @Positive
    String getBaseName();

    @Positive
    void appendTo(StringBuilder sb);

    @Positive
    boolean isAssignable(@Nullable Method m1, @Nullable Method m2);
    @Positive
}

// CFWR semantic augmentation - variant 1
