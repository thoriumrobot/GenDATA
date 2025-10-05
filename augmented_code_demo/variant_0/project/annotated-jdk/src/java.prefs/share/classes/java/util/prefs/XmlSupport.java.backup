/*
 * Copyright (c) 2002, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util.prefs;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.*;
import java.io.*;
import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import org.xml.sax.*;
import org.w3c.dom.*;
import static java.nio.charset.StandardCharsets.UTF_8;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
class XmlSupport {

    static void export(OutputStream os, final Preferences p, boolean subTree) throws IOException, BackingStoreException;

    static void importPreferences(InputStream is) throws IOException, InvalidPreferencesFormatException;

    static void exportMap(OutputStream os, Map<String, String> map) throws IOException;

    static void importMap(InputStream is, Map<String, String> m) throws IOException, InvalidPreferencesFormatException;

    private static class Resolver implements EntityResolver {

        public InputSource resolveEntity(String pid, String sid) throws SAXException;
    }

    private static class EH implements ErrorHandler {

        public void error(SAXParseException x) throws SAXException;

        public void fatalError(SAXParseException x) throws SAXException;

        public void warning(SAXParseException x) throws SAXException;
    }
}
