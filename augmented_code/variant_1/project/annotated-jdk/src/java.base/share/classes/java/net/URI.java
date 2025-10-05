/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.net;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.CoderResult;
    @Positive
import java.nio.charset.CodingErrorAction;
    @Positive
import java.nio.charset.CharacterCodingException;
    @Positive
import java.nio.file.Path;
    @Positive
import java.text.Normalizer;
    @Positive
import jdk.internal.access.JavaNetUriAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import java.lang.Character;
    @Positive
import java.lang.NullPointerException;

    @Positive
public final class URI implements Comparable<URI>, Serializable {

    @Positive
    public URI(String str) throws URISyntaxException {
    @Positive
    }

    @Positive
    public URI(String scheme, String userInfo, String host, int port, String path, String query, String fragment) throws URISyntaxException {
    @Positive
    }

    @Positive
    public URI(String scheme, String authority, String path, String query, String fragment) throws URISyntaxException {
    @Positive
    }

    @Positive
    public URI(String scheme, String host, String path, String fragment) throws URISyntaxException {
    @Positive
    }

    @Positive
    public URI(String scheme, String ssp, String fragment) throws URISyntaxException {
    @Positive
    }

    @Positive
    public static URI create(String str);

    @Positive
    public URI parseServerAuthority() throws URISyntaxException;

    @Positive
    public URI normalize();

    @Positive
    public URI resolve(URI uri);

    @Positive
    public URI resolve(String str);

    @Positive
    public URI relativize(URI uri);

    @Positive
    public URL toURL() throws MalformedURLException;

    @Positive
    public String getScheme();

    @Positive
    public boolean isAbsolute();

    @Positive
    public boolean isOpaque();

    @Positive
    public String getRawSchemeSpecificPart();

    @Positive
    public String getSchemeSpecificPart();

    @Positive
    public String getRawAuthority();

    @Positive
    public String getAuthority();

    @Positive
    public String getRawUserInfo();

    @Positive
    public String getUserInfo();

    @Positive
    public String getHost();

    @Positive
    public int getPort();

    @Positive
    public String getRawPath();

    @Positive
    public String getPath();

    @Positive
    public String getRawQuery();

    @Positive
    public String getQuery();

    @Positive
    public String getRawFragment();

    @Positive
    public String getFragment();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object ob);

    @Positive
    public int hashCode();

    @Positive
    public int compareTo(URI that);

    @Positive
    public String toString();

    @Positive
    public String toASCIIString();

    @Positive
    private class Parser {

    @Positive
        void parse(boolean rsa) throws URISyntaxException;
    @Positive
    }
    @Positive
}
