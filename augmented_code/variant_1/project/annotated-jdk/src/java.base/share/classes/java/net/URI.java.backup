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
package java.net;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.File;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.CharacterCodingException;
import java.nio.file.Path;
import java.text.Normalizer;
import jdk.internal.access.JavaNetUriAccess;
import jdk.internal.access.SharedSecrets;
import sun.nio.cs.UTF_8;
import java.lang.Character;
import java.lang.NullPointerException;

public final class URI implements Comparable<URI>, Serializable {

    public URI(String str) throws URISyntaxException {
    }

    public URI(String scheme, String userInfo, String host, int port, String path, String query, String fragment) throws URISyntaxException {
    }

    public URI(String scheme, String authority, String path, String query, String fragment) throws URISyntaxException {
    }

    public URI(String scheme, String host, String path, String fragment) throws URISyntaxException {
    }

    public URI(String scheme, String ssp, String fragment) throws URISyntaxException {
    }

    public static URI create(String str);

    public URI parseServerAuthority() throws URISyntaxException;

    public URI normalize();

    public URI resolve(URI uri);

    public URI resolve(String str);

    public URI relativize(URI uri);

    public URL toURL() throws MalformedURLException;

    public String getScheme();

    public boolean isAbsolute();

    public boolean isOpaque();

    public String getRawSchemeSpecificPart();

    public String getSchemeSpecificPart();

    public String getRawAuthority();

    public String getAuthority();

    public String getRawUserInfo();

    public String getUserInfo();

    public String getHost();

    public int getPort();

    public String getRawPath();

    public String getPath();

    public String getRawQuery();

    public String getQuery();

    public String getRawFragment();

    public String getFragment();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object ob);

    public int hashCode();

    public int compareTo(URI that);

    public String toString();

    public String toASCIIString();

    private class Parser {

        void parse(boolean rsa) throws URISyntaxException;
    }
}
