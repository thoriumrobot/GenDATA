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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Arrays;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Inet6Address extends InetAddress {

    @Positive
    private static class Inet6AddressHolder {

    @Positive
        void setAddr(byte[] addr);

    @Positive
        void init(byte[] addr, int scope_id);

    @Positive
        void init(byte[] addr, NetworkInterface nif) throws UnknownHostException;

    @Positive
        String getHostAddress();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        boolean isIPv4CompatibleAddress();

    @Positive
        boolean isMulticastAddress();

    @Positive
        boolean isAnyLocalAddress();

    @Positive
        boolean isLoopbackAddress();

    @Positive
        boolean isLinkLocalAddress();

    @Positive
        boolean isSiteLocalAddress();

    @Positive
        boolean isMCGlobal();

    @Positive
        boolean isMCNodeLocal();

    @Positive
        boolean isMCLinkLocal();

    @Positive
        boolean isMCSiteLocal();

    @Positive
        boolean isMCOrgLocal();
    @Positive
    }

    @Positive
    public static Inet6Address getByAddress(String host, byte[] addr, NetworkInterface nif) throws UnknownHostException;

    @Positive
    public static Inet6Address getByAddress(String host, byte[] addr, int scope_id) throws UnknownHostException;

    @Positive
    @Override
    @Positive
    public boolean isMulticastAddress();

    @Positive
    @Override
    @Positive
    public boolean isAnyLocalAddress();

    @Positive
    @Override
    @Positive
    public boolean isLoopbackAddress();

    @Positive
    @Override
    @Positive
    public boolean isLinkLocalAddress();

    @Positive
    static boolean isLinkLocalAddress(byte[] ipaddress);

    @Positive
    @Override
    @Positive
    public boolean isSiteLocalAddress();

    @Positive
    static boolean isSiteLocalAddress(byte[] ipaddress);

    @Positive
    @Override
    @Positive
    public boolean isMCGlobal();

    @Positive
    @Override
    @Positive
    public boolean isMCNodeLocal();

    @Positive
    @Override
    @Positive
    public boolean isMCLinkLocal();

    @Positive
    @Override
    @Positive
    public boolean isMCSiteLocal();

    @Positive
    @Override
    @Positive
    public boolean isMCOrgLocal();

    @Positive
    @Override
    @Positive
    public byte[] getAddress();

    @Positive
    byte[] addressBytes();

    @Positive
    public int getScopeId();

    @Positive
    public NetworkInterface getScopedInterface();

    @Positive
    @Override
    @Positive
    public String getHostAddress();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public boolean isIPv4CompatibleAddress();

    @Positive
    static String numericToTextFormat(byte[] src);
    @Positive
}
