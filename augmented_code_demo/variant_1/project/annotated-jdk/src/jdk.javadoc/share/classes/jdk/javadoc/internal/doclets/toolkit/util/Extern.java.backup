/*
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

import org.checkerframework.dataflow.qual.Pure;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.tools.Diagnostic;
import javax.tools.Diagnostic.Kind;
import javax.tools.DocumentationTool;
import jdk.javadoc.doclet.Reporter;
import jdk.javadoc.internal.doclets.toolkit.AbstractDoclet;
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
import jdk.javadoc.internal.doclets.toolkit.Resources;

public class Extern {

    private static class Item {

        @Override
        public String toString();
    }

    public Extern(BaseConfiguration configuration) {
    }

    @Pure
    public boolean isExternal(Element element);

    @Pure
    public boolean isModule(String elementName);

    public DocLink getExternalLink(Element element, DocPath relativepath, String filename);

    public DocLink getExternalLink(Element element, DocPath relativepath, String filename, String memberName);

    public boolean link(String url, Reporter reporter) throws DocFileIOException;

    public boolean link(String url, String elemlisturl, Reporter reporter) throws DocFileIOException;

    public void checkPlatformLinks(String linkPlatformProperties, Reporter reporter);

    private static class Fault extends Exception {
    }

    @Pure
    public boolean isUrl(String urlCandidate);
}
