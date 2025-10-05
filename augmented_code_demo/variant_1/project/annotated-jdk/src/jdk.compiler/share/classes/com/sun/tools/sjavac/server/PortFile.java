/*
    @Positive
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.sjavac.server;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.File;
    @Positive
import java.io.FileNotFoundException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.RandomAccessFile;
    @Positive
import java.nio.channels.ClosedChannelException;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import java.nio.channels.FileLock;
    @Positive
import java.nio.channels.FileLockInterruptionException;
    @Positive
import java.util.concurrent.Semaphore;
    @Positive
import com.sun.tools.javac.util.Assert;
    @Positive
import com.sun.tools.sjavac.Log;
    @Positive
import com.sun.tools.sjavac.client.PortFileInaccessibleException;

    @Positive
public class PortFile {

    @Positive
    public PortFile(String fn) {
    @Positive
    }

    @Positive
    public void lock() throws IOException, InterruptedException;

    @Positive
    public void getValues();

    @Positive
    @Pure
    @Positive
    public boolean containsPortInfo();

    @Positive
    public int getPort();

    @Positive
    public long getCookie();

    @Positive
    public void setValues(int port, long cookie) throws IOException;

    @Positive
    public void delete() throws IOException, InterruptedException;

    @Positive
    public boolean exists() throws IOException;

    @Positive
    public boolean markedForStop() throws IOException;

    @Positive
    public void unlock() throws IOException;

    @Positive
    public void waitForValidValues() throws IOException, InterruptedException;

    @Positive
    public boolean stillMyValues() throws IOException, FileNotFoundException, InterruptedException;

    @Positive
    public String getFilename();
    @Positive
}

// CFWR semantic augmentation - variant 1
