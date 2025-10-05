/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.InputStream;
import java.io.File;
import java.io.OutputStream;
import java.util.Hashtable;
import java.util.Objects;
import sun.net.util.IPAddressUtil;
import sun.net.www.ParseUtil;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class URLStreamHandler {

    public URLStreamHandler() {
    }

    protected abstract URLConnection openConnection(URL u) throws IOException;

    protected URLConnection openConnection(URL u, Proxy p) throws IOException;

    protected void parseURL(URL u, String spec, int start, int limit);

    protected int getDefaultPort();

    protected boolean equals(URL u1, URL u2);

    protected int hashCode(URL u);

    protected boolean sameFile(URL u1, URL u2);

    protected InetAddress getHostAddress(URL u);

    protected boolean hostsEqual(URL u1, URL u2);

    protected String toExternalForm(URL u);

    protected void setURL(URL u, String protocol, String host, int port, String authority, String userInfo, String path, String query, String ref);

    @Deprecated
    protected void setURL(URL u, String protocol, String host, int port, String file, String ref);
}
