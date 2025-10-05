/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.parsers;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import com.sun.org.apache.xerces.internal.jaxp.DocumentBuilderFactoryImpl;
import javax.xml.validation.Schema;

@AnnotatedFor("nullness")
public abstract class DocumentBuilderFactory {

    protected DocumentBuilderFactory() {
    }

    public static DocumentBuilderFactory newDefaultNSInstance();

    public static DocumentBuilderFactory newNSInstance();

    public static DocumentBuilderFactory newNSInstance(String factoryClassName, @Nullable ClassLoader classLoader);

    public static DocumentBuilderFactory newDefaultInstance();

    public static DocumentBuilderFactory newInstance();

    public static DocumentBuilderFactory newInstance(String factoryClassName, @Nullable ClassLoader classLoader);

    public abstract DocumentBuilder newDocumentBuilder() throws ParserConfigurationException;

    public void setNamespaceAware(boolean awareness);

    public void setValidating(boolean validating);

    public void setIgnoringElementContentWhitespace(boolean whitespace);

    public void setExpandEntityReferences(boolean expandEntityRef);

    public void setIgnoringComments(boolean ignoreComments);

    public void setCoalescing(boolean coalescing);

    public boolean isNamespaceAware();

    public boolean isValidating();

    public boolean isIgnoringElementContentWhitespace();

    public boolean isExpandEntityReferences();

    public boolean isIgnoringComments();

    public boolean isCoalescing();

    public abstract void setAttribute(String name, Object value) throws IllegalArgumentException;

    public abstract Object getAttribute(String name) throws IllegalArgumentException;

    public abstract void setFeature(String name, boolean value) throws ParserConfigurationException;

    public abstract boolean getFeature(String name) throws ParserConfigurationException;

    @CFComment("nullness: this.getClass().getPackage() is non-null as this class is in the `parsers` package")
    @SuppressWarnings({ "nullness" })
    @Nullable
    public Schema getSchema();

    @CFComment("nullness: this.getClass().getPackage() is non-null as this class is in the `parsers` package")
    @SuppressWarnings({ "nullness" })
    public void setSchema(@Nullable Schema schema);

    public void setXIncludeAware(final boolean state);

    @CFComment("nullness: this.getClass().getPackage() is non-null as this class is in the `parsers` package")
    @SuppressWarnings({ "nullness" })
    public boolean isXIncludeAware();
}
