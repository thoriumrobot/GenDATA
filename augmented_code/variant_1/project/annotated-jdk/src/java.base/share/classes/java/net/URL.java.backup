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

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.spi.URLStreamHandlerProvider;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Hashtable;
import java.io.InvalidObjectException;
import java.io.ObjectStreamException;
import java.io.ObjectStreamField;
import java.io.ObjectInputStream.GetField;
import java.util.Iterator;
import java.util.Locale;
import java.util.NoSuchElementException;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import jdk.internal.access.JavaNetURLAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.misc.VM;
import sun.net.util.IPAddressUtil;
import sun.security.util.SecurityConstants;
import sun.security.action.GetPropertyAction;

@AnnotatedFor("nullness")
public final class URL implements java.io.Serializable {

    public URL(String protocol, @Nullable String host, int port, String file) throws MalformedURLException {
    }

    public URL(String protocol, @Nullable String host, String file) throws MalformedURLException {
    }

    public URL(String protocol, @Nullable String host, int port, String file, @Nullable URLStreamHandler handler) throws MalformedURLException {
    }

    public URL(String spec) throws MalformedURLException {
    }

    public URL(@Nullable URL context, String spec) throws MalformedURLException {
    }

    public URL(@Nullable URL context, String spec, @Nullable URLStreamHandler handler) throws MalformedURLException {
    }

    static URL fromURI(URI uri) throws MalformedURLException;

    void set(String protocol, String host, int port, String authority, String userInfo, String path, String query, String ref);

    synchronized InetAddress getHostAddress();

    public String getQuery();

    public String getPath();

    public String getUserInfo();

    public String getAuthority();

    public int getPort();

    public int getDefaultPort();

    public String getProtocol();

    public String getHost();

    public String getFile();

    public String getRef();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public synchronized int hashCode();

    public boolean sameFile(URL other);

    public String toString();

    public String toExternalForm();

    public URI toURI() throws URISyntaxException;

    public URLConnection openConnection() throws java.io.IOException;

    public URLConnection openConnection(Proxy proxy) throws java.io.IOException;

    public final InputStream openStream() throws java.io.IOException;

    public final Object getContent() throws java.io.IOException;

    @Nullable
    public final Object getContent(Class<?>[] classes) throws java.io.IOException;

    public static void setURLStreamHandlerFactory(URLStreamHandlerFactory fac);

    private static class DefaultFactory implements URLStreamHandlerFactory {

        public URLStreamHandler createURLStreamHandler(String protocol);
    }

    static String toLowerCase(String protocol);

    static boolean isOverrideable(String protocol);

    static URLStreamHandler getURLStreamHandler(String protocol);

    boolean isBuiltinStreamHandler(URLStreamHandler handler);
}

final class UrlDeserializedState {

    public UrlDeserializedState(String protocol, String host, int port, String authority, String file, String ref, int hashCode) {
    }

    String getProtocol();

    String getHost();

    String getAuthority();

    int getPort();

    String getFile();

    String getRef();

    int getHashCode();

    String reconstituteUrlString();
}
