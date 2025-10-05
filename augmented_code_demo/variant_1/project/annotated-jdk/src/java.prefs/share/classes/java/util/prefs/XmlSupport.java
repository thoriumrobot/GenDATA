/*
    @Positive
 * Copyright (c) 2002, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util.prefs;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.*;
    @Positive
import java.io.*;
    @Positive
import javax.xml.parsers.*;
    @Positive
import javax.xml.transform.*;
    @Positive
import javax.xml.transform.dom.*;
    @Positive
import javax.xml.transform.stream.*;
    @Positive
import org.xml.sax.*;
    @Positive
import org.w3c.dom.*;
    @Positive
import static java.nio.charset.StandardCharsets.UTF_8;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class XmlSupport {

    @Positive
    static void export(OutputStream os, final Preferences p, boolean subTree) throws IOException, BackingStoreException;

    @Positive
    static void importPreferences(InputStream is) throws IOException, InvalidPreferencesFormatException;

    @Positive
    static void exportMap(OutputStream os, Map<String, String> map) throws IOException;

    @Positive
    static void importMap(InputStream is, Map<String, String> m) throws IOException, InvalidPreferencesFormatException;

    @Positive
    private static class Resolver implements EntityResolver {

    @Positive
        public InputSource resolveEntity(String pid, String sid) throws SAXException;
    @Positive
    }

    @Positive
    private static class EH implements ErrorHandler {

    @Positive
        public void error(SAXParseException x) throws SAXException;

    @Positive
        public void fatalError(SAXParseException x) throws SAXException;

    @Positive
        public void warning(SAXParseException x) throws SAXException;
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
