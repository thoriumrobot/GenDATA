/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class HttpConnectSocketImpl {
/*
    @Positive
 * Copyright (c) 2010, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;

    @Positive
@AnnotatedFor("nullness")
    @Positive
@SuppressWarnings("removal")
    @Positive
class HttpConnectSocketImpl extends DelegatingSocketImpl {

    @Positive
    @Override
    @Positive
    protected void connect(String host, int port) throws IOException;

    @Positive
    @Override
    @Positive
    protected void connect(InetAddress address, int port) throws IOException;

    @Positive
    @Override
    @Positive
    protected void connect(SocketAddress endpoint, int timeout) throws IOException;

    @Positive
    @Override
    @Positive
    protected void listen(int backlog);

    @Positive
    @Override
    @Positive
    protected void accept(SocketImpl s);

    @Positive
    @Override
    @Positive
    void reset();

    @Positive
    @Override
    @Positive
    public void setOption(int opt, Object val) throws SocketException;

    @Positive
    @Override
    @Positive
    protected InetAddress getInetAddress();

    @Positive
    @Override
    @Positive
    protected int getPort();
    @Positive
}

}