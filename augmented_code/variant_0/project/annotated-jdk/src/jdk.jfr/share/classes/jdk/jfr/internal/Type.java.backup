/*
 * Copyright (c) 2016, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jfr.internal;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import jdk.jfr.AnnotationElement;
import jdk.jfr.Event;
import jdk.jfr.SettingControl;
import jdk.jfr.ValueDescriptor;

public class Type implements Comparable<Type> {

    public static final String SUPER_TYPE_ANNOTATION;

    public static final String SUPER_TYPE_SETTING;

    public static final String SUPER_TYPE_EVENT;

    public static final String EVENT_NAME_PREFIX;

    public static final String TYPES_PREFIX;

    public static final String SETTINGS_PREFIX;

    public Type(String javaTypeName, String superType, long typeId) {
    }

    static boolean isDefinedByJVM(long id);

    public static long getTypeId(Class<?> clazz);

    static Collection<Type> getKnownTypes();

    public static boolean isValidJavaIdentifier(String identifier);

    public static boolean isValidJavaFieldType(String name);

    public static Type getKnownType(String typeName);

    static boolean isKnownType(Class<?> type);

    public static Type getKnownType(Class<?> clazz);

    public String getName();

    public String getLogName();

    public ValueDescriptor getField(String name);

    public List<ValueDescriptor> getFields();

    public boolean isSimpleType();

    public boolean isDefinedByJVM();

    public void add(ValueDescriptor valueDescriptor);

    public int indexOf(String name);

    void trimFields();

    void setAnnotations(List<AnnotationElement> annotations);

    public String getSuperType();

    public long getId();

    public String getLabel();

    public List<AnnotationElement> getAnnotationElements();

    public <T> T getAnnotation(Class<? extends java.lang.annotation.Annotation> clazz);

    public String getDescription();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object object);

    @Override
    public int compareTo(Type that);

    void log(String action, LogTag logTag, LogLevel level);

    @Override
    public String toString();

    public void setRemove(boolean remove);

    public boolean getRemove();

    public void setId(long id);
}
