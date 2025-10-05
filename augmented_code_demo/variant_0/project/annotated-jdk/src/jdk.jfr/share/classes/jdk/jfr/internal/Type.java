/*
    @Positive
 * Copyright (c) 2016, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jfr.internal;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import jdk.jfr.AnnotationElement;
    @Positive
import jdk.jfr.Event;
    @Positive
import jdk.jfr.SettingControl;
    @Positive
import jdk.jfr.ValueDescriptor;

    @Positive
public class Type implements Comparable<Type> {

    @Positive
    public static final String SUPER_TYPE_ANNOTATION;

    @Positive
    public static final String SUPER_TYPE_SETTING;

    @Positive
    public static final String SUPER_TYPE_EVENT;

    @Positive
    public static final String EVENT_NAME_PREFIX;

    @Positive
    public static final String TYPES_PREFIX;

    @Positive
    public static final String SETTINGS_PREFIX;

    @Positive
    public Type(String javaTypeName, String superType, long typeId) {
    @Positive
    }

    @Positive
    static boolean isDefinedByJVM(long id);

    @Positive
    public static long getTypeId(Class<?> clazz);

    @Positive
    static Collection<Type> getKnownTypes();

    @Positive
    public static boolean isValidJavaIdentifier(String identifier);

    @Positive
    public static boolean isValidJavaFieldType(String name);

    @Positive
    public static Type getKnownType(String typeName);

    @Positive
    static boolean isKnownType(Class<?> type);

    @Positive
    public static Type getKnownType(Class<?> clazz);

    @Positive
    public String getName();

    @Positive
    public String getLogName();

    @Positive
    public ValueDescriptor getField(String name);

    @Positive
    public List<ValueDescriptor> getFields();

    @Positive
    public boolean isSimpleType();

    @Positive
    public boolean isDefinedByJVM();

    @Positive
    public void add(ValueDescriptor valueDescriptor);

    @Positive
    public int indexOf(String name);

    @Positive
    void trimFields();

    @Positive
    void setAnnotations(List<AnnotationElement> annotations);

    @Positive
    public String getSuperType();

    @Positive
    public long getId();

    @Positive
    public String getLabel();

    @Positive
    public List<AnnotationElement> getAnnotationElements();

    @Positive
    public <T> T getAnnotation(Class<? extends java.lang.annotation.Annotation> clazz);

    @Positive
    public String getDescription();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object object);

    @Positive
    @Override
    @Positive
    public int compareTo(Type that);

    @Positive
    void log(String action, LogTag logTag, LogLevel level);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public void setRemove(boolean remove);

    @Positive
    public boolean getRemove();

    @Positive
    public void setId(long id);
    @Positive
}

// CFWR semantic augmentation - variant 0
