/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.File;
    @Positive
import java.io.OutputStream;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Objects;
    @Positive
import sun.net.util.IPAddressUtil;
    @Positive
import sun.net.www.ParseUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class URLStreamHandler {

    @Positive
    public URLStreamHandler() {
    @Positive
    }

    @Positive
    protected abstract URLConnection openConnection(URL u) throws IOException;

    @Positive
    protected URLConnection openConnection(URL u, Proxy p) throws IOException;

    @Positive
    protected void parseURL(URL u, String spec, int start, int limit);

    @Positive
    protected int getDefaultPort();

    @Positive
    protected boolean equals(URL u1, URL u2);

    @Positive
    protected int hashCode(URL u);

    @Positive
    protected boolean sameFile(URL u1, URL u2);

    @Positive
    protected InetAddress getHostAddress(URL u);

    @Positive
    protected boolean hostsEqual(URL u1, URL u2);

    @Positive
    protected String toExternalForm(URL u);

    @Positive
    protected void setURL(URL u, String protocol, String host, int port, String authority, String userInfo, String path, String query, String ref);

    @Positive
    @Deprecated
    @Positive
    protected void setURL(URL u, String protocol, String host, int port, String file, String ref);
    @Positive
}

// CFWR semantic augmentation - variant 1
