/*
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.openmbean;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.management.Descriptor;
import javax.management.DescriptorRead;
import javax.management.ImmutableDescriptor;
import javax.management.MBeanAttributeInfo;
import com.sun.jmx.remote.util.EnvHelp;
import sun.reflect.misc.MethodUtil;
import sun.reflect.misc.ReflectUtil;

public class OpenMBeanAttributeInfoSupport extends MBeanAttributeInfo implements OpenMBeanAttributeInfo {

    public OpenMBeanAttributeInfoSupport(String name, String description, OpenType<?> openType, boolean isReadable, boolean isWritable, boolean isIs) {
    }

    public OpenMBeanAttributeInfoSupport(String name, String description, OpenType<?> openType, boolean isReadable, boolean isWritable, boolean isIs, Descriptor descriptor) {
    }

    public <T> OpenMBeanAttributeInfoSupport(String name, String description, OpenType<T> openType, boolean isReadable, boolean isWritable, boolean isIs, T defaultValue) throws OpenDataException {
    }

    public <T> OpenMBeanAttributeInfoSupport(String name, String description, OpenType<T> openType, boolean isReadable, boolean isWritable, boolean isIs, T defaultValue, T[] legalValues) throws OpenDataException {
    }

    public <T> OpenMBeanAttributeInfoSupport(String name, String description, OpenType<T> openType, boolean isReadable, boolean isWritable, boolean isIs, T defaultValue, Comparable<T> minValue, Comparable<T> maxValue) throws OpenDataException {
    }

    static void check(OpenMBeanParameterInfo info) throws OpenDataException;

    @SuppressWarnings({ "unchecked", "rawtypes" })
    static int compare(Object x, Object y);

    static <T> Descriptor makeDescriptor(OpenType<T> openType, T defaultValue, T[] legalValues, Comparable<T> minValue, Comparable<T> maxValue);

    static <T> Descriptor makeDescriptor(OpenType<T> openType, T defaultValue, Set<T> legalValues, Comparable<T> minValue, Comparable<T> maxValue);

    static <T> T valueFrom(Descriptor d, String name, OpenType<T> openType);

    static <T> Set<T> valuesFrom(Descriptor d, String name, OpenType<T> openType);

    static <T> Comparable<?> comparableValueFrom(Descriptor d, String name, OpenType<T> openType);

    @SuppressWarnings("unchecked")
    static <T> T cast(Object x);

    public OpenType<?> getOpenType();

    public Object getDefaultValue();

    public Set<?> getLegalValues();

    public Comparable<?> getMinValue();

    public Comparable<?> getMaxValue();

    public boolean hasDefaultValue();

    public boolean hasLegalValues();

    public boolean hasMinValue();

    public boolean hasMaxValue();

    public boolean isValue(Object obj);

    @SuppressWarnings({ "unchecked", "rawtypes" })
    static boolean isValue(OpenMBeanParameterInfo info, Object obj);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    static boolean equal(OpenMBeanParameterInfo x1, OpenMBeanParameterInfo x2);

    public int hashCode();

    static int hashCode(OpenMBeanParameterInfo info);

    public String toString();

    static String toString(OpenMBeanParameterInfo info);
}
