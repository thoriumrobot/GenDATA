/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.net.HttpURLConnection;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLConnection;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Properties;
    @Positive
import java.util.TreeMap;
    @Positive
import javax.lang.model.SourceVersion;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.tools.Diagnostic;
    @Positive
import javax.tools.Diagnostic.Kind;
    @Positive
import javax.tools.DocumentationTool;
    @Positive
import jdk.javadoc.doclet.Reporter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.AbstractDoclet;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Resources;

    @Positive
public class Extern {

    @Positive
    private static class Item {

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public Extern(BaseConfiguration configuration) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean isExternal(Element element);

    @Positive
    @Pure
    @Positive
    public boolean isModule(String elementName);

    @Positive
    public DocLink getExternalLink(Element element, DocPath relativepath, String filename);

    @Positive
    public DocLink getExternalLink(Element element, DocPath relativepath, String filename, String memberName);

    @Positive
    public boolean link(String url, Reporter reporter) throws DocFileIOException;

    @Positive
    public boolean link(String url, String elemlisturl, Reporter reporter) throws DocFileIOException;

    @Positive
    public void checkPlatformLinks(String linkPlatformProperties, Reporter reporter);

    @Positive
    private static class Fault extends Exception {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean isUrl(String urlCandidate);
    @Positive
}
