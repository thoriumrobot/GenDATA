/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.management;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.StreamCorruptedException;
import java.io.Serializable;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Map;
import java.util.WeakHashMap;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Objects;
import static javax.management.ImmutableDescriptor.nonNullDescriptor;

public class MBeanInfo implements Cloneable, Serializable, DescriptorRead {

    public MBeanInfo(String className, String description, MBeanAttributeInfo[] attributes, MBeanConstructorInfo[] constructors, MBeanOperationInfo[] operations, MBeanNotificationInfo[] notifications) throws IllegalArgumentException {
    }

    public MBeanInfo(String className, String description, MBeanAttributeInfo[] attributes, MBeanConstructorInfo[] constructors, MBeanOperationInfo[] operations, MBeanNotificationInfo[] notifications, Descriptor descriptor) throws IllegalArgumentException {
    }

    @Override
    public Object clone();

    public String getClassName();

    public String getDescription();

    public MBeanAttributeInfo[] getAttributes();

    public MBeanOperationInfo[] getOperations();

    public MBeanConstructorInfo[] getConstructors();

    public MBeanNotificationInfo[] getNotifications();

    public Descriptor getDescriptor();

    @Override
    public String toString();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    @SuppressWarnings("removal")
    static boolean arrayGettersSafe(Class<?> subclass, Class<?> immutableClass);

    private static class ArrayGettersSafeAction implements PrivilegedAction<Boolean> {

        public Boolean run();
    }
}
