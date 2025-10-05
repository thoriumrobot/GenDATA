/*
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.sjavac.server;

import org.checkerframework.dataflow.qual.Pure;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.FileLockInterruptionException;
import java.util.concurrent.Semaphore;
import com.sun.tools.javac.util.Assert;
import com.sun.tools.sjavac.Log;
import com.sun.tools.sjavac.client.PortFileInaccessibleException;

public class PortFile {

    public PortFile(String fn) {
    }

    public void lock() throws IOException, InterruptedException;

    public void getValues();

    @Pure
    public boolean containsPortInfo();

    public int getPort();

    public long getCookie();

    public void setValues(int port, long cookie) throws IOException;

    public void delete() throws IOException, InterruptedException;

    public boolean exists() throws IOException;

    public boolean markedForStop() throws IOException;

    public void unlock() throws IOException;

    public void waitForValidValues() throws IOException, InterruptedException;

    public boolean stillMyValues() throws IOException, FileNotFoundException, InterruptedException;

    public String getFilename();
}
