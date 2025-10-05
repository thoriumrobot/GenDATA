/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.rmi.server;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.net.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class RMISocketFactory implements RMIClientSocketFactory, RMIServerSocketFactory {

    @Positive
    public RMISocketFactory() {
    @Positive
    }

    @Positive
    public abstract Socket createSocket(String host, int port) throws IOException;

    @Positive
    public abstract ServerSocket createServerSocket(int port) throws IOException;

    @Positive
    public synchronized static void setSocketFactory(RMISocketFactory fac) throws IOException;

    @Positive
    public synchronized static RMISocketFactory getSocketFactory();

    @Positive
    public synchronized static RMISocketFactory getDefaultSocketFactory();

    @Positive
    public synchronized static void setFailureHandler(RMIFailureHandler fh);

    @Positive
    public synchronized static RMIFailureHandler getFailureHandler();
    @Positive
}
