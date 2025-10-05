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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.net.spi.URLStreamHandlerProvider;
    @Positive
import java.nio.file.Path;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Hashtable;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.ObjectInputStream.GetField;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.ServiceConfigurationError;
    @Positive
import java.util.ServiceLoader;
    @Positive
import jdk.internal.access.JavaNetURLAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import sun.net.util.IPAddressUtil;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public final class URL implements java.io.Serializable {

    @Positive
    public URL(String protocol, @Nullable String host, int port, String file) throws MalformedURLException {
    @Positive
    }

    @Positive
    public URL(String protocol, @Nullable String host, String file) throws MalformedURLException {
    @Positive
    }

    @Positive
    public URL(String protocol, @Nullable String host, int port, String file, @Nullable URLStreamHandler handler) throws MalformedURLException {
    @Positive
    }

    @Positive
    public URL(String spec) throws MalformedURLException {
    @Positive
    }

    @Positive
    public URL(@Nullable URL context, String spec) throws MalformedURLException {
    @Positive
    }

    @Positive
    public URL(@Nullable URL context, String spec, @Nullable URLStreamHandler handler) throws MalformedURLException {
    @Positive
    }

    @Positive
    static URL fromURI(URI uri) throws MalformedURLException;

    @Positive
    void set(String protocol, String host, int port, String authority, String userInfo, String path, String query, String ref);

    @Positive
    synchronized InetAddress getHostAddress();

    @Positive
    public String getQuery();

    @Positive
    public String getPath();

    @Positive
    public String getUserInfo();

    @Positive
    public String getAuthority();

    @Positive
    public int getPort();

    @Positive
    public int getDefaultPort();

    @Positive
    public String getProtocol();

    @Positive
    public String getHost();

    @Positive
    public String getFile();

    @Positive
    public String getRef();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public synchronized int hashCode();

    @Positive
    public boolean sameFile(URL other);

    @Positive
    public String toString();

    @Positive
    public String toExternalForm();

    @Positive
    public URI toURI() throws URISyntaxException;

    @Positive
    public URLConnection openConnection() throws java.io.IOException;

    @Positive
    public URLConnection openConnection(Proxy proxy) throws java.io.IOException;

    @Positive
    public final InputStream openStream() throws java.io.IOException;

    @Positive
    public final Object getContent() throws java.io.IOException;

    @Positive
    @Nullable
    @Positive
    public final Object getContent(Class<?>[] classes) throws java.io.IOException;

    @Positive
    public static void setURLStreamHandlerFactory(URLStreamHandlerFactory fac);

    @Positive
    private static class DefaultFactory implements URLStreamHandlerFactory {

    @Positive
        public URLStreamHandler createURLStreamHandler(String protocol);
    @Positive
    }

    @Positive
    static String toLowerCase(String protocol);

    @Positive
    static boolean isOverrideable(String protocol);

    @Positive
    static URLStreamHandler getURLStreamHandler(String protocol);

    @Positive
    boolean isBuiltinStreamHandler(URLStreamHandler handler);
    @Positive
}

    @Positive
final class UrlDeserializedState {

    @Positive
    public UrlDeserializedState(String protocol, String host, int port, String authority, String file, String ref, int hashCode) {
    @Positive
    }

    @Positive
    String getProtocol();

    @Positive
    String getHost();

    @Positive
    String getAuthority();

    @Positive
    int getPort();

    @Positive
    String getFile();

    @Positive
    String getRef();

    @Positive
    int getHashCode();

    @Positive
    String reconstituteUrlString();
    @Positive
}

// CFWR semantic augmentation - variant 0
