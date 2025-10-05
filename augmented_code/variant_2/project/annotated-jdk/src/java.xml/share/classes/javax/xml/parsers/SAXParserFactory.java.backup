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
import com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl;
import javax.xml.validation.Schema;
import org.xml.sax.SAXException;
import org.xml.sax.SAXNotRecognizedException;
import org.xml.sax.SAXNotSupportedException;

@CFComment("nullness")
public abstract class SAXParserFactory {

    protected SAXParserFactory() {
    }

    public static SAXParserFactory newDefaultNSInstance();

    public static SAXParserFactory newNSInstance();

    public static SAXParserFactory newNSInstance(String factoryClassName, ClassLoader classLoader);

    public static SAXParserFactory newDefaultInstance();

    public static SAXParserFactory newInstance();

    public static SAXParserFactory newInstance(String factoryClassName, @Nullable ClassLoader classLoader);

    public abstract SAXParser newSAXParser() throws ParserConfigurationException, SAXException;

    public void setNamespaceAware(boolean awareness);

    public void setValidating(boolean validating);

    public boolean isNamespaceAware();

    public boolean isValidating();

    public abstract void setFeature(String name, boolean value) throws ParserConfigurationException, SAXNotRecognizedException, SAXNotSupportedException;

    public abstract boolean getFeature(String name) throws ParserConfigurationException, SAXNotRecognizedException, SAXNotSupportedException;

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
