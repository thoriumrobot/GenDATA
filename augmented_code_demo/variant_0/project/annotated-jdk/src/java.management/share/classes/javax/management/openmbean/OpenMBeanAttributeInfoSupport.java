/*
    @Positive
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.openmbean;

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
import java.lang.reflect.Array;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import javax.management.Descriptor;
    @Positive
import javax.management.DescriptorRead;
    @Positive
import javax.management.ImmutableDescriptor;
    @Positive
import javax.management.MBeanAttributeInfo;
    @Positive
import com.sun.jmx.remote.util.EnvHelp;
    @Positive
import sun.reflect.misc.MethodUtil;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
public class OpenMBeanAttributeInfoSupport extends MBeanAttributeInfo implements OpenMBeanAttributeInfo {

    @Positive
    public OpenMBeanAttributeInfoSupport(String name, String description, OpenType<?> openType, boolean isReadable, boolean isWritable, boolean isIs) {
    @Positive
    }

    @Positive
    public OpenMBeanAttributeInfoSupport(String name, String description, OpenType<?> openType, boolean isReadable, boolean isWritable, boolean isIs, Descriptor descriptor) {
    @Positive
    }

    @Positive
    public <T> OpenMBeanAttributeInfoSupport(String name, String description, OpenType<T> openType, boolean isReadable, boolean isWritable, boolean isIs, T defaultValue) throws OpenDataException {
    @Positive
    }

    @Positive
    public <T> OpenMBeanAttributeInfoSupport(String name, String description, OpenType<T> openType, boolean isReadable, boolean isWritable, boolean isIs, T defaultValue, T[] legalValues) throws OpenDataException {
    @Positive
    }

    @Positive
    public <T> OpenMBeanAttributeInfoSupport(String name, String description, OpenType<T> openType, boolean isReadable, boolean isWritable, boolean isIs, T defaultValue, Comparable<T> minValue, Comparable<T> maxValue) throws OpenDataException {
    @Positive
    }

    @Positive
    static void check(OpenMBeanParameterInfo info) throws OpenDataException;

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    static int compare(Object x, Object y);

    @Positive
    static <T> Descriptor makeDescriptor(OpenType<T> openType, T defaultValue, T[] legalValues, Comparable<T> minValue, Comparable<T> maxValue);

    @Positive
    static <T> Descriptor makeDescriptor(OpenType<T> openType, T defaultValue, Set<T> legalValues, Comparable<T> minValue, Comparable<T> maxValue);

    @Positive
    static <T> T valueFrom(Descriptor d, String name, OpenType<T> openType);

    @Positive
    static <T> Set<T> valuesFrom(Descriptor d, String name, OpenType<T> openType);

    @Positive
    static <T> Comparable<?> comparableValueFrom(Descriptor d, String name, OpenType<T> openType);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <T> T cast(Object x);

    @Positive
    public OpenType<?> getOpenType();

    @Positive
    public Object getDefaultValue();

    @Positive
    public Set<?> getLegalValues();

    @Positive
    public Comparable<?> getMinValue();

    @Positive
    public Comparable<?> getMaxValue();

    @Positive
    public boolean hasDefaultValue();

    @Positive
    public boolean hasLegalValues();

    @Positive
    public boolean hasMinValue();

    @Positive
    public boolean hasMaxValue();

    @Positive
    public boolean isValue(Object obj);

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    static boolean isValue(OpenMBeanParameterInfo info, Object obj);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    static boolean equal(OpenMBeanParameterInfo x1, OpenMBeanParameterInfo x2);

    @Positive
    public int hashCode();

    @Positive
    static int hashCode(OpenMBeanParameterInfo info);

    @Positive
    public String toString();

    @Positive
    static String toString(OpenMBeanParameterInfo info);
    @Positive
}

// CFWR semantic augmentation - variant 0
