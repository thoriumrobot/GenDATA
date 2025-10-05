/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.geom.AffineTransform;
import java.awt.image.BaseMultiResolutionImage;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.DirectColorModel;
import java.awt.image.MultiResolutionImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.awt.peer.RobotPeer;
import sun.awt.AWTPermissions;
import sun.awt.ComponentFactory;
import sun.awt.SunToolkit;
import sun.awt.image.SunWritableRaster;
import sun.java2d.SunGraphicsEnvironment;
import static sun.java2d.SunGraphicsEnvironment.toDeviceSpace;
import static sun.java2d.SunGraphicsEnvironment.toDeviceSpaceAbs;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Robot {

    public Robot() throws AWTException {
    }

    public Robot(GraphicsDevice screen) throws AWTException {
    }

    public synchronized void mouseMove(int x, int y);

    public synchronized void mousePress(int buttons);

    public synchronized void mouseRelease(int buttons);

    public synchronized void mouseWheel(int wheelAmt);

    public synchronized void keyPress(int keycode);

    public synchronized void keyRelease(int keycode);

    public synchronized Color getPixelColor(int x, int y);

    public synchronized BufferedImage createScreenCapture(Rectangle screenRect);

    public synchronized MultiResolutionImage createMultiResolutionScreenCapture(Rectangle screenRect);

    public synchronized boolean isAutoWaitForIdle();

    public synchronized void setAutoWaitForIdle(boolean isOn);

    public synchronized int getAutoDelay();

    public synchronized void setAutoDelay(int ms);

    public void delay(int ms);

    public synchronized void waitForIdle();

    @Override
    public synchronized String toString();
}
