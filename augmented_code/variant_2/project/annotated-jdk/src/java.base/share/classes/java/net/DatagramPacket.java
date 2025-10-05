/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class DatagramPacket {

    @Positive
    public DatagramPacket(byte[] buf, int offset, int length) {
    @Positive
    }

    @Positive
    public DatagramPacket(byte[] buf, int length) {
    @Positive
    }

    @Positive
    public DatagramPacket(byte[] buf, int offset, int length, InetAddress address, int port) {
    @Positive
    }

    @Positive
    public DatagramPacket(byte[] buf, int offset, int length, SocketAddress address) {
    @Positive
    }

    @Positive
    public DatagramPacket(byte[] buf, int length, InetAddress address, int port) {
    @Positive
    }

    @Positive
    public DatagramPacket(byte[] buf, int length, SocketAddress address) {
    @Positive
    }

    @Positive
    public synchronized InetAddress getAddress();

    @Positive
    public synchronized int getPort();

    @Positive
    public synchronized byte[] getData();

    @Positive
    public synchronized int getOffset();

    @Positive
    public synchronized int getLength();

    @Positive
    public synchronized void setData(byte[] buf, int offset, int length);

    @Positive
    public synchronized void setAddress(InetAddress iaddr);

    @Positive
    public synchronized void setPort(int iport);

    @Positive
    public synchronized void setSocketAddress(SocketAddress address);

    @Positive
    public synchronized SocketAddress getSocketAddress();

    @Positive
    public synchronized void setData(byte[] buf);

    @Positive
    public synchronized void setLength(int length);
    @Positive
}
