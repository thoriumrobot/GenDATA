/*
    @Positive
 * Copyright (c) 2000, 2017, Oracle and/or its affiliates. All rights reserved.
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
package javax.management;

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
import java.io.Serializable;
    @Positive
import java.lang.management.MemoryUsage;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Locale;
    @Positive
import java.util.ResourceBundle;
    @Positive
import javax.management.openmbean.CompositeData;
    @Positive
import javax.management.openmbean.OpenMBeanAttributeInfoSupport;
    @Positive
import javax.management.openmbean.OpenMBeanOperationInfoSupport;
    @Positive
import javax.management.openmbean.OpenMBeanParameterInfoSupport;
    @Positive
import javax.management.openmbean.OpenType;

    @Positive
public interface Descriptor extends Serializable, Cloneable {

    @Positive
    public Object getFieldValue(String fieldName) throws RuntimeOperationsException;

    @Positive
    public void setField(String fieldName, Object fieldValue) throws RuntimeOperationsException;

    @Positive
    public String[] getFields();

    @Positive
    public String[] getFieldNames();

    @Positive
    public Object[] getFieldValues(String... fieldNames);

    @Positive
    public void removeField(String fieldName);

    @Positive
    public void setFields(String[] fieldNames, Object[] fieldValues) throws RuntimeOperationsException;

    @Positive
    public Object clone() throws RuntimeOperationsException;

    @Positive
    public boolean isValid() throws RuntimeOperationsException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0
