/*
 * Copyright (c) 2004, 2019, Oracle and/or its affiliates. All rights reserved.
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
import com.sun.jmx.mbeanserver.Util;
import java.io.InvalidObjectException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

public class ImmutableDescriptor implements Descriptor {

    public static final ImmutableDescriptor EMPTY_DESCRIPTOR;

    public ImmutableDescriptor(String[] fieldNames, Object[] fieldValues) {
    }

    public ImmutableDescriptor(String... fields) {
    }

    public ImmutableDescriptor(Map<String, ?> fields) {
    }

    public static ImmutableDescriptor union(Descriptor... descriptors);

    public final Object getFieldValue(String fieldName);

    public final String[] getFields();

    public final Object[] getFieldValues(String... fieldNames);

    public final String[] getFieldNames();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    @Override
    public String toString();

    public boolean isValid();

    @Override
    public Descriptor clone();

    public final void setFields(String[] fieldNames, Object[] fieldValues) throws RuntimeOperationsException;

    public final void setField(String fieldName, Object fieldValue) throws RuntimeOperationsException;

    public final void removeField(String fieldName);

    static Descriptor nonNullDescriptor(Descriptor d);
}
