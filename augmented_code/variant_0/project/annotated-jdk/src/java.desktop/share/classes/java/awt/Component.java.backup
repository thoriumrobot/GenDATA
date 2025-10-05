/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.guieffect.qual.UIType;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.applet.Applet;
import java.awt.dnd.DropTarget;
import java.awt.event.ActionEvent;
import java.awt.event.AdjustmentEvent;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.HierarchyBoundsListener;
import java.awt.event.HierarchyEvent;
import java.awt.event.HierarchyListener;
import java.awt.event.InputEvent;
import java.awt.event.InputMethodEvent;
import java.awt.event.InputMethodListener;
import java.awt.event.ItemEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.event.PaintEvent;
import java.awt.event.TextEvent;
import java.awt.im.InputContext;
import java.awt.im.InputMethodRequests;
import java.awt.image.BufferStrategy;
import java.awt.image.ColorModel;
import java.awt.image.ImageObserver;
import java.awt.image.ImageProducer;
import java.awt.image.VolatileImage;
import java.awt.peer.ComponentPeer;
import java.awt.peer.ContainerPeer;
import java.awt.peer.LightweightPeer;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.beans.Transient;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serial;
import java.io.Serializable;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.util.Collections;
import java.util.EventListener;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.Vector;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleComponent;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.accessibility.AccessibleSelection;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import javax.swing.JComponent;
import javax.swing.JRootPane;
import sun.awt.AWTAccessor;
import sun.awt.AppContext;
import sun.awt.ComponentFactory;
import sun.awt.ConstrainableGraphics;
import sun.awt.EmbeddedFrame;
import sun.awt.RequestFocusController;
import sun.awt.SubRegionShowable;
import sun.awt.SunToolkit;
import sun.awt.dnd.SunDropTargetEvent;
import sun.awt.im.CompositionArea;
import sun.awt.image.VSyncedBSManager;
import sun.font.FontManager;
import sun.font.FontManagerFactory;
import sun.font.SunFontManager;
import sun.java2d.SunGraphics2D;
import sun.java2d.SunGraphicsEnvironment;
import sun.java2d.pipe.Region;
import sun.java2d.pipe.hw.ExtendedBufferCapabilities;
import sun.security.action.GetPropertyAction;
import sun.swing.SwingAccessor;
import sun.util.logging.PlatformLogger;
import static sun.java2d.pipe.hw.ExtendedBufferCapabilities.VSyncType.VSYNC_DEFAULT;
import static sun.java2d.pipe.hw.ExtendedBufferCapabilities.VSyncType.VSYNC_ON;

@AnnotatedFor({ "guieffect", "interning", "nullness" })
@UsesObjectEquals
@UIType
public abstract class Component implements ImageObserver, MenuContainer, Serializable {

    static class AWTTreeLock {
    }

    public static final float TOP_ALIGNMENT;

    public static final float CENTER_ALIGNMENT;

    public static final float BOTTOM_ALIGNMENT;

    public static final float LEFT_ALIGNMENT;

    public static final float RIGHT_ALIGNMENT;

    Object getObjectLock();

    @SuppressWarnings("removal")
    final AccessControlContext getAccessControlContext();

    public enum BaselineResizeBehavior {

        CONSTANT_ASCENT, CONSTANT_DESCENT, CENTER_OFFSET, OTHER
    }

    int getBoundsOp();

    void setBoundsOp(int op);

    protected Component() {
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
    void initializeFocusTraversalKeys();

    @Nullable
    String constructComponentName();

    public String getName();

    public void setName(String name);

    @Nullable
    public Container getParent();

    @Nullable
    final Container getParent_NoClientCode();

    @Nullable
    Container getContainer();

    public synchronized void setDropTarget(DropTarget dt);

    @Nullable
    public synchronized DropTarget getDropTarget();

    @Nullable
    public GraphicsConfiguration getGraphicsConfiguration();

    @Nullable
    final GraphicsConfiguration getGraphicsConfiguration_NoClientCode();

    void setGraphicsConfiguration(GraphicsConfiguration gc);

    final boolean updateGraphicsData(GraphicsConfiguration gc);

    boolean updateChildGraphicsData(GraphicsConfiguration gc);

    void checkGD(String stringID);

    public final Object getTreeLock();

    final void checkTreeLock();

    public Toolkit getToolkit();

    final Toolkit getToolkitImpl();

    final ComponentFactory getComponentFactory();

    public boolean isValid();

    public boolean isDisplayable();

    @Transient
    public boolean isVisible();

    final boolean isVisible_NoClientCode();

    boolean isRecursivelyVisible();

    Point pointRelativeToComponent(Point absolute);

    @Nullable
    Component findUnderMouseInWindow(PointerInfo pi);

    public Point getMousePosition() throws HeadlessException;

    boolean isSameOrAncestorOf(Component comp, boolean allowChildren);

    public boolean isShowing();

    public boolean isEnabled();

    final boolean isEnabledImpl();

    public void setEnabled(boolean b);

    @Deprecated
    public void enable();

    @Deprecated
    public void enable(boolean b);

    @Deprecated
    public void disable();

    public boolean isDoubleBuffered();

    public void enableInputMethods(boolean enable);

    public void setVisible(boolean b);

    @Deprecated
    public void show();

    @Deprecated
    public void show(boolean b);

    boolean containsFocus();

    void clearMostRecentFocusOwnerOnHide();

    void clearCurrentFocusCycleRootOnHide();

    @Deprecated
    public void hide();

    @Transient
    @Nullable
    public Color getForeground();

    public void setForeground(@Nullable Color c);

    public boolean isForegroundSet();

    @Transient
    @Nullable
    public Color getBackground();

    public void setBackground(@Nullable Color c);

    public boolean isBackgroundSet();

    @Transient
    @Nullable
    public Font getFont();

    @Nullable
    final Font getFont_NoClientCode();

    public void setFont(Font f);

    public boolean isFontSet();

    public Locale getLocale();

    public void setLocale(Locale l);

    public ColorModel getColorModel();

    public Point getLocation();

    public Point getLocationOnScreen();

    final Point getLocationOnScreen_NoTreeLock();

    @Deprecated
    public Point location();

    public void setLocation(int x, int y);

    @Deprecated
    public void move(int x, int y);

    public void setLocation(Point p);

    public Dimension getSize();

    @Deprecated
    public Dimension size();

    public void setSize(int width, int height);

    @Deprecated
    public void resize(int width, int height);

    public void setSize(Dimension d);

    @Deprecated
    public void resize(Dimension d);

    public Rectangle getBounds();

    @Deprecated
    public Rectangle bounds();

    public void setBounds(int x, int y, int width, int height);

    @Deprecated
    public void reshape(int x, int y, int width, int height);

    public void setBounds(Rectangle r);

    public int getX();

    public int getY();

    public int getWidth();

    public int getHeight();

    public Rectangle getBounds(Rectangle rv);

    public Dimension getSize(Dimension rv);

    public Point getLocation(Point rv);

    public boolean isOpaque();

    public boolean isLightweight();

    public void setPreferredSize(@Nullable Dimension preferredSize);

    public boolean isPreferredSizeSet();

    public Dimension getPreferredSize();

    @Deprecated
    public Dimension preferredSize();

    public void setMinimumSize(@Nullable Dimension minimumSize);

    public boolean isMinimumSizeSet();

    public Dimension getMinimumSize();

    @Deprecated
    public Dimension minimumSize();

    public void setMaximumSize(@Nullable Dimension maximumSize);

    public boolean isMaximumSizeSet();

    public Dimension getMaximumSize();

    public float getAlignmentX();

    public float getAlignmentY();

    public int getBaseline(int width, int height);

    public BaselineResizeBehavior getBaselineResizeBehavior();

    public void doLayout();

    @Deprecated
    public void layout();

    public void validate();

    public void invalidate();

    void invalidateParent();

    final void invalidateIfValid();

    public void revalidate();

    final void revalidateSynchronously();

    @Nullable
    public Graphics getGraphics();

    @Nullable
    final Graphics getGraphics_NoClientCode();

    public FontMetrics getFontMetrics(Font font);

    public void setCursor(@Nullable Cursor cursor);

    final void updateCursorImmediately();

    public Cursor getCursor();

    final Cursor getCursor_NoClientCode();

    public boolean isCursorSet();

    public void paint(Graphics g);

    public void update(Graphics g);

    public void paintAll(Graphics g);

    void lightweightPaint(Graphics g);

    void paintHeavyweightComponents(Graphics g);

    @SafeEffect
    public void repaint();

    @SafeEffect
    public void repaint(long tm);

    @SafeEffect
    public void repaint(int x, int y, int width, int height);

    @SafeEffect
    public void repaint(long tm, int x, int y, int width, int height);

    public void print(Graphics g);

    public void printAll(Graphics g);

    void lightweightPrint(Graphics g);

    void printHeavyweightComponents(Graphics g);

    public boolean imageUpdate(Image img, int infoflags, int x, int y, int w, int h);

    public Image createImage(ImageProducer producer);

    @Nullable
    public Image createImage(int width, int height);

    @Nullable
    public VolatileImage createVolatileImage(int width, int height);

    @Nullable
    public VolatileImage createVolatileImage(int width, int height, ImageCapabilities caps) throws AWTException;

    public boolean prepareImage(Image image, ImageObserver observer);

    public boolean prepareImage(Image image, int width, int height, ImageObserver observer);

    public int checkImage(Image image, ImageObserver observer);

    public int checkImage(Image image, int width, int height, ImageObserver observer);

    void createBufferStrategy(int numBuffers);

    void createBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException;

    private class ProxyCapabilities extends ExtendedBufferCapabilities {
    }

    BufferStrategy getBufferStrategy();

    Image getBackBuffer();

    protected class FlipBufferStrategy extends BufferStrategy {

        protected int numBuffers;

        protected BufferCapabilities caps;

        protected Image drawBuffer;

        protected VolatileImage drawVBuffer;

        protected boolean validatedContents;

        @SuppressWarnings("removal")
        protected FlipBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException {
        }

        protected void createBuffers(int numBuffers, BufferCapabilities caps) throws AWTException;

        protected Image getBackBuffer();

        protected void flip(BufferCapabilities.FlipContents flipAction);

        void flipSubRegion(int x1, int y1, int x2, int y2, BufferCapabilities.FlipContents flipAction);

        protected void destroyBuffers();

        public BufferCapabilities getCapabilities();

        public Graphics getDrawGraphics();

        protected void revalidate();

        public boolean contentsLost();

        public boolean contentsRestored();

        public void show();

        void showSubRegion(int x1, int y1, int x2, int y2);

        public void dispose();
    }

    protected class BltBufferStrategy extends BufferStrategy {

        protected BufferCapabilities caps;

        protected VolatileImage[] backBuffers;

        protected boolean validatedContents;

        protected int width;

        protected int height;

        protected BltBufferStrategy(int numBuffers, BufferCapabilities caps) {
        }

        public void dispose();

        protected void createBackBuffers(int numBuffers);

        public BufferCapabilities getCapabilities();

        public Graphics getDrawGraphics();

        Image getBackBuffer();

        public void show();

        void showSubRegion(int x1, int y1, int x2, int y2);

        protected void revalidate();

        void revalidate(boolean checkSize);

        public boolean contentsLost();

        public boolean contentsRestored();
    }

    private class FlipSubRegionBufferStrategy extends FlipBufferStrategy implements SubRegionShowable {

        protected FlipSubRegionBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException {
        }

        public void show(int x1, int y1, int x2, int y2);

        public boolean showIfNotLost(int x1, int y1, int x2, int y2);
    }

    private class BltSubRegionBufferStrategy extends BltBufferStrategy implements SubRegionShowable {

        protected BltSubRegionBufferStrategy(int numBuffers, BufferCapabilities caps) {
        }

        public void show(int x1, int y1, int x2, int y2);

        public boolean showIfNotLost(int x1, int y1, int x2, int y2);
    }

    private class SingleBufferStrategy extends BufferStrategy {

        public SingleBufferStrategy(BufferCapabilities caps) {
        }

        public BufferCapabilities getCapabilities();

        public Graphics getDrawGraphics();

        public boolean contentsLost();

        public boolean contentsRestored();

        public void show();
    }

    public void setIgnoreRepaint(boolean ignoreRepaint);

    public boolean getIgnoreRepaint();

    public boolean contains(int x, int y);

    @Deprecated
    public boolean inside(int x, int y);

    public boolean contains(Point p);

    @Nullable
    public Component getComponentAt(int x, int y);

    @Deprecated
    @Nullable
    public Component locate(int x, int y);

    @Nullable
    public Component getComponentAt(Point p);

    @Deprecated
    public void deliverEvent(Event e);

    public final void dispatchEvent(AWTEvent e);

    @SuppressWarnings("deprecation")
    void dispatchEventImpl(AWTEvent e);

    void autoProcessMouseWheel(MouseWheelEvent e);

    @SuppressWarnings("deprecation")
    boolean dispatchMouseWheelToAncestor(MouseWheelEvent e);

    boolean areInputMethodsEnabled();

    boolean eventEnabled(AWTEvent e);

    boolean eventTypeEnabled(int type);

    @Deprecated
    public boolean postEvent(Event e);

    public synchronized void addComponentListener(ComponentListener l);

    public synchronized void removeComponentListener(ComponentListener l);

    public synchronized ComponentListener[] getComponentListeners();

    public synchronized void addFocusListener(FocusListener l);

    public synchronized void removeFocusListener(FocusListener l);

    public synchronized FocusListener[] getFocusListeners();

    public void addHierarchyListener(HierarchyListener l);

    public void removeHierarchyListener(HierarchyListener l);

    public synchronized HierarchyListener[] getHierarchyListeners();

    public void addHierarchyBoundsListener(HierarchyBoundsListener l);

    public void removeHierarchyBoundsListener(HierarchyBoundsListener l);

    int numListening(long mask);

    int countHierarchyMembers();

    int createHierarchyEvents(int id, Component changed, Container changedParent, long changeFlags, boolean enabledOnToolkit);

    public synchronized HierarchyBoundsListener[] getHierarchyBoundsListeners();

    void adjustListeningChildrenOnParent(long mask, int num);

    public synchronized void addKeyListener(KeyListener l);

    public synchronized void removeKeyListener(KeyListener l);

    public synchronized KeyListener[] getKeyListeners();

    public synchronized void addMouseListener(MouseListener l);

    public synchronized void removeMouseListener(MouseListener l);

    public synchronized MouseListener[] getMouseListeners();

    public synchronized void addMouseMotionListener(MouseMotionListener l);

    public synchronized void removeMouseMotionListener(MouseMotionListener l);

    public synchronized MouseMotionListener[] getMouseMotionListeners();

    public synchronized void addMouseWheelListener(MouseWheelListener l);

    public synchronized void removeMouseWheelListener(MouseWheelListener l);

    public synchronized MouseWheelListener[] getMouseWheelListeners();

    public synchronized void addInputMethodListener(InputMethodListener l);

    public synchronized void removeInputMethodListener(InputMethodListener l);

    public synchronized InputMethodListener[] getInputMethodListeners();

    @SuppressWarnings("unchecked")
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    public InputMethodRequests getInputMethodRequests();

    @Nullable
    public InputContext getInputContext();

    protected final void enableEvents(long eventsToEnable);

    protected final void disableEvents(long eventsToDisable);

    final boolean isCoalescingEnabled();

    protected AWTEvent coalesceEvents(AWTEvent existingEvent, AWTEvent newEvent);

    protected void processEvent(AWTEvent e);

    protected void processComponentEvent(ComponentEvent e);

    protected void processFocusEvent(FocusEvent e);

    protected void processKeyEvent(KeyEvent e);

    protected void processMouseEvent(MouseEvent e);

    protected void processMouseMotionEvent(MouseEvent e);

    protected void processMouseWheelEvent(MouseWheelEvent e);

    boolean postsOldMouseEvents();

    protected void processInputMethodEvent(InputMethodEvent e);

    protected void processHierarchyEvent(HierarchyEvent e);

    protected void processHierarchyBoundsEvent(HierarchyEvent e);

    @Deprecated
    public boolean handleEvent(Event evt);

    @Deprecated
    public boolean mouseDown(Event evt, int x, int y);

    @Deprecated
    public boolean mouseDrag(Event evt, int x, int y);

    @Deprecated
    public boolean mouseUp(Event evt, int x, int y);

    @Deprecated
    public boolean mouseMove(Event evt, int x, int y);

    @Deprecated
    public boolean mouseEnter(Event evt, int x, int y);

    @Deprecated
    public boolean mouseExit(Event evt, int x, int y);

    @Deprecated
    public boolean keyDown(Event evt, int key);

    @Deprecated
    public boolean keyUp(Event evt, int key);

    @Deprecated
    public boolean action(Event evt, Object what);

    public void addNotify();

    public void removeNotify();

    @Deprecated
    public boolean gotFocus(Event evt, Object what);

    @Deprecated
    public boolean lostFocus(Event evt, Object what);

    @Deprecated
    public boolean isFocusTraversable();

    public boolean isFocusable();

    public void setFocusable(boolean focusable);

    final boolean isFocusTraversableOverridden();

    public void setFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    public Set<AWTKeyStroke> getFocusTraversalKeys(int id);

    final void setFocusTraversalKeys_NoIDCheck(int id, Set<? extends AWTKeyStroke> keystrokes);

    final Set<AWTKeyStroke> getFocusTraversalKeys_NoIDCheck(int id);

    public boolean areFocusTraversalKeysSet(int id);

    public void setFocusTraversalKeysEnabled(boolean focusTraversalKeysEnabled);

    public boolean getFocusTraversalKeysEnabled();

    public void requestFocus();

    public void requestFocus(FocusEvent.Cause cause);

    protected boolean requestFocus(boolean temporary);

    protected boolean requestFocus(boolean temporary, FocusEvent.Cause cause);

    public boolean requestFocusInWindow();

    public boolean requestFocusInWindow(FocusEvent.Cause cause);

    protected boolean requestFocusInWindow(boolean temporary);

    boolean requestFocusInWindow(boolean temporary, FocusEvent.Cause cause);

    final boolean requestFocusHelper(boolean temporary, boolean focusedWindowChangeAllowed);

    final boolean requestFocusHelper(boolean temporary, boolean focusedWindowChangeAllowed, FocusEvent.Cause cause);

    private static class DummyRequestFocusController implements RequestFocusController {

        public boolean acceptRequestFocus(Component from, Component to, boolean temporary, boolean focusedWindowChangeAllowed, FocusEvent.Cause cause);
    }

    static synchronized void setRequestFocusController(RequestFocusController requestController);

    @Nullable
    public Container getFocusCycleRootAncestor();

    public boolean isFocusCycleRoot(Container container);

    Container getTraversalRoot();

    public void transferFocus();

    @Deprecated
    public void nextFocus();

    boolean transferFocus(boolean clearOnFailure);

    @SuppressWarnings("removal")
    final Component getNextFocusCandidate();

    public void transferFocusBackward();

    boolean transferFocusBackward(boolean clearOnFailure);

    public void transferFocusUpCycle();

    public boolean hasFocus();

    public boolean isFocusOwner();

    void setAutoFocusTransferOnDisposal(boolean value);

    boolean isAutoFocusTransferOnDisposal();

    public void add(PopupMenu popup);

    @SuppressWarnings("unchecked")
    public void remove(MenuComponent popup);

    protected String paramString();

    public String toString();

    public void list();

    public void list(PrintStream out);

    public void list(PrintStream out, int indent);

    public void list(PrintWriter out);

    public void list(PrintWriter out, int indent);

    final Container getNativeContainer();

    public void addPropertyChangeListener(PropertyChangeListener listener);

    public void removePropertyChangeListener(PropertyChangeListener listener);

    public PropertyChangeListener[] getPropertyChangeListeners();

    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    public void removePropertyChangeListener(String propertyName, PropertyChangeListener listener);

    public PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    protected void firePropertyChange(String propertyName, @Nullable Object oldValue, @Nullable Object newValue);

    protected void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);

    protected void firePropertyChange(String propertyName, int oldValue, int newValue);

    public void firePropertyChange(String propertyName, byte oldValue, byte newValue);

    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    public void firePropertyChange(String propertyName, short oldValue, short newValue);

    public void firePropertyChange(String propertyName, long oldValue, long newValue);

    public void firePropertyChange(String propertyName, float oldValue, float newValue);

    public void firePropertyChange(String propertyName, double oldValue, double newValue);

    public void setComponentOrientation(ComponentOrientation o);

    public ComponentOrientation getComponentOrientation();

    public void applyComponentOrientation(ComponentOrientation orientation);

    final boolean canBeFocusOwner();

    final boolean canBeFocusOwnerRecursively();

    final void relocateComponent();

    @Nullable
    Window getContainingWindow();

    @SuppressWarnings("serial")
    protected AccessibleContext accessibleContext;

    public AccessibleContext getAccessibleContext();

    protected abstract class AccessibleAWTComponent extends AccessibleContext implements Serializable, AccessibleComponent {

        protected AccessibleAWTComponent() {
        }

        @SuppressWarnings("serial")
        protected ComponentListener accessibleAWTComponentHandler;

        @SuppressWarnings("serial")
        protected FocusListener accessibleAWTFocusHandler;

        protected class AccessibleAWTComponentHandler implements ComponentListener, Serializable {

            protected AccessibleAWTComponentHandler() {
            }

            public void componentHidden(ComponentEvent e);

            public void componentShown(ComponentEvent e);

            public void componentMoved(ComponentEvent e);

            public void componentResized(ComponentEvent e);
        }

        protected class AccessibleAWTFocusHandler implements FocusListener, Serializable {

            protected AccessibleAWTFocusHandler() {
            }

            public void focusGained(FocusEvent event);

            public void focusLost(FocusEvent event);
        }

        public void addPropertyChangeListener(PropertyChangeListener listener);

        public void removePropertyChangeListener(PropertyChangeListener listener);

        public String getAccessibleName();

        public String getAccessibleDescription();

        public AccessibleRole getAccessibleRole();

        public AccessibleStateSet getAccessibleStateSet();

        public Accessible getAccessibleParent();

        public int getAccessibleIndexInParent();

        public int getAccessibleChildrenCount();

        public Accessible getAccessibleChild(int i);

        public Locale getLocale();

        public AccessibleComponent getAccessibleComponent();

        public Color getBackground();

        public void setBackground(Color c);

        public Color getForeground();

        public void setForeground(Color c);

        public Cursor getCursor();

        public void setCursor(Cursor cursor);

        public Font getFont();

        public void setFont(Font f);

        public FontMetrics getFontMetrics(Font f);

        public boolean isEnabled();

        public void setEnabled(boolean b);

        public boolean isVisible();

        public void setVisible(boolean b);

        public boolean isShowing();

        public boolean contains(Point p);

        public Point getLocationOnScreen();

        public Point getLocation();

        public void setLocation(Point p);

        public Rectangle getBounds();

        public void setBounds(Rectangle r);

        public Dimension getSize();

        public void setSize(Dimension d);

        public Accessible getAccessibleAt(Point p);

        public boolean isFocusTraversable();

        public void requestFocus();

        public void addFocusListener(FocusListener l);

        public void removeFocusListener(FocusListener l);
    }

    int getAccessibleIndexInParent();

    AccessibleStateSet getAccessibleStateSet();

    static boolean isInstanceOf(Object obj, String className);

    final boolean areBoundsValid();

    void applyCompoundShape(Region shape);

    Point getLocationOnWindow();

    final Region getNormalShape();

    Region getOpaqueShape();

    final int getSiblingIndexAbove();

    @Nullable
    final ComponentPeer getHWPeerAboveMe();

    final int getSiblingIndexBelow();

    final boolean isNonOpaqueForMixing();

    void applyCurrentShape();

    final void subtractAndApplyShape(Region s);

    final void subtractAndApplyShapeBelowMe();

    void mixOnShowing();

    void mixOnHiding(boolean isLightweight);

    void mixOnReshaping();

    void mixOnZOrderChanging(int oldZorder, int newZorder);

    void mixOnValidating();

    final boolean isMixingNeeded();

    public void setMixingCutoutShape(@Nullable Shape shape);

    void updateZOrder();
}
