/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.parsers;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl;
    @Positive
import javax.xml.validation.Schema;
    @Positive
import org.xml.sax.SAXException;
    @Positive
import org.xml.sax.SAXNotRecognizedException;
    @Positive
import org.xml.sax.SAXNotSupportedException;

    @Positive
@CFComment("nullness")
    @Positive
public abstract class SAXParserFactory {

    @Positive
    protected SAXParserFactory() {
    @Positive
    }

    @Positive
    public static SAXParserFactory newDefaultNSInstance();

    @Positive
    public static SAXParserFactory newNSInstance();

    @Positive
    public static SAXParserFactory newNSInstance(String factoryClassName, ClassLoader classLoader);

    @Positive
    public static SAXParserFactory newDefaultInstance();

    @Positive
    public static SAXParserFactory newInstance();

    @Positive
    public static SAXParserFactory newInstance(String factoryClassName, @Nullable ClassLoader classLoader);

    @Positive
    public abstract SAXParser newSAXParser() throws ParserConfigurationException, SAXException;

    @Positive
    public void setNamespaceAware(boolean awareness);

    @Positive
    public void setValidating(boolean validating);

    @Positive
    public boolean isNamespaceAware();

    @Positive
    public boolean isValidating();

    @Positive
    public abstract void setFeature(String name, boolean value) throws ParserConfigurationException, SAXNotRecognizedException, SAXNotSupportedException;

    @Positive
    public abstract boolean getFeature(String name) throws ParserConfigurationException, SAXNotRecognizedException, SAXNotSupportedException;

    @Positive
    @CFComment("nullness: this.getClass().getPackage() is non-null as this class is in the `parsers` package")
    @Positive
    @SuppressWarnings({ "nullness" })
    @Positive
    @Nullable
    @Positive
    public Schema getSchema();

    @Positive
    @CFComment("nullness: this.getClass().getPackage() is non-null as this class is in the `parsers` package")
    @Positive
    @SuppressWarnings({ "nullness" })
    @Positive
    public void setSchema(@Nullable Schema schema);

    @Positive
    public void setXIncludeAware(final boolean state);

    @Positive
    @CFComment("nullness: this.getClass().getPackage() is non-null as this class is in the `parsers` package")
    @Positive
    @SuppressWarnings({ "nullness" })
    @Positive
    public boolean isXIncludeAware();
    @Positive
}

// CFWR semantic augmentation - variant 0
