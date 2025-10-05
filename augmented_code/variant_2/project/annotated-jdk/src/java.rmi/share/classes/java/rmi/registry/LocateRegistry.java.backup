/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.rmi.registry;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.rmi.RemoteException;
import java.rmi.server.ObjID;
import java.rmi.server.RMIClientSocketFactory;
import java.rmi.server.RMIServerSocketFactory;
import java.rmi.server.RemoteRef;
import java.rmi.server.UnicastRemoteObject;
import sun.rmi.registry.RegistryImpl;
import sun.rmi.server.UnicastRef2;
import sun.rmi.server.UnicastRef;
import sun.rmi.server.Util;
import sun.rmi.transport.LiveRef;
import sun.rmi.transport.tcp.TCPEndpoint;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class LocateRegistry {

    public static Registry getRegistry() throws RemoteException;

    public static Registry getRegistry(int port) throws RemoteException;

    public static Registry getRegistry(String host) throws RemoteException;

    public static Registry getRegistry(String host, int port) throws RemoteException;

    public static Registry getRegistry(String host, int port, RMIClientSocketFactory csf) throws RemoteException;

    public static Registry createRegistry(int port) throws RemoteException;

    public static Registry createRegistry(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf) throws RemoteException;
}
