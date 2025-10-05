/*
 * Copyright (c) 1996, 2014, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.ref.Reference;
import java.lang.reflect.Method;
import java.util.Map.Entry;
import com.sun.beans.introspect.PropertyInfo;

public class IndexedPropertyDescriptor extends PropertyDescriptor {

    public IndexedPropertyDescriptor(String propertyName, Class<?> beanClass) throws IntrospectionException {
    }

    public IndexedPropertyDescriptor(String propertyName, Class<?> beanClass, String readMethodName, String writeMethodName, String indexedReadMethodName, String indexedWriteMethodName) throws IntrospectionException {
    }

    public IndexedPropertyDescriptor(String propertyName, Method readMethod, Method writeMethod, Method indexedReadMethod, Method indexedWriteMethod) throws IntrospectionException {
    }

    public synchronized Method getIndexedReadMethod();

    public synchronized void setIndexedReadMethod(Method readMethod) throws IntrospectionException;

    public synchronized Method getIndexedWriteMethod();

    public synchronized void setIndexedWriteMethod(Method writeMethod) throws IntrospectionException;

    public synchronized Class<?> getIndexedPropertyType();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    void updateGenericsFor(Class<?> type);

    public int hashCode();

    void appendTo(StringBuilder sb);
}
