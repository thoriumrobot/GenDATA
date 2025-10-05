/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.org.apache.xerces.internal.impl;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.utils.XMLSecurityManager;
    @Positive
import com.sun.org.apache.xerces.internal.utils.XMLSecurityPropertyManager;
    @Positive
import com.sun.xml.internal.stream.StaxEntityResolverWrapper;
    @Positive
import java.util.HashMap;
    @Positive
import javax.xml.XMLConstants;
    @Positive
import javax.xml.catalog.CatalogFeatures;
    @Positive
import javax.xml.stream.XMLInputFactory;
    @Positive
import javax.xml.stream.XMLOutputFactory;
    @Positive
import javax.xml.stream.XMLResolver;
    @Positive
import jdk.xml.internal.JdkConstants;
    @Positive
import jdk.xml.internal.JdkProperty;
    @Positive
import jdk.xml.internal.JdkXmlUtils;

    @Positive
public class PropertyManager {

    @Positive
    public static final String STAX_NOTATIONS;

    @Positive
    public static final String STAX_ENTITIES;

    @Positive
    public static final int CONTEXT_READER;

    @Positive
    public static final int CONTEXT_WRITER;

    @Positive
    public PropertyManager(int context) {
    @Positive
    }

    @Positive
    public PropertyManager(PropertyManager propertyManager) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean containsProperty(String property);

    @Positive
    public Object getProperty(String property);

    @Positive
    public void setProperty(String property, Object value);

    @Positive
    public String toString();
    @Positive
}
